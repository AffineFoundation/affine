"""Admit orphaned durable trainer exports to the existing cache lifecycle.

This CPU operator is invoked before capacity admission and after normal ACK
retirement. It never discovers arbitrary paths or removes optimizer state.
Each candidate needs a ROOT-signed inventory, complete R2 readback, and exact
local bytes; ordinary successful jobs continue using trainer_cache_lifecycle.
"""
import hashlib
import json
import os
import re
from pathlib import Path

VERSION = 'authenticated-obsolete-trainer-export-catalog-v1'


class StaleCatalogError(ValueError):
    """Refresh the current anchor; never relax the unchanged disk reserve."""



def validate_catalog(envelope, authority, workspace):
    from subnet.backend_jobs import signed, file_map
    value = signed(envelope, authority)
    if set(value) != {'version', 'workspace', 'current_ack', 'candidates'} or value['version'] != VERSION:
        raise ValueError('exact obsolete export catalog')
    root = Path(workspace).absolute()
    if str(root) != value['workspace'] or root != root.resolve():
        raise ValueError('canonical owned trainer workspace')
    ack = signed(value['current_ack'], authority)
    if ack.get('version') != 'durable-original-trainer-cache-ACK-v1' or ack.get('authority_state_committed') is not True:
        raise ValueError('original committed trainer ACK')
    rows = value['candidates']
    if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
        raise ValueError('bounded explicit obsolete export catalog')
    seen = set()
    for row in rows:
        if set(row) != {'directory', 'original_job', 'publication_job', 'checkpoint', 'readback'}:
            raise ValueError('exact obsolete export member')
        job = signed(row['original_job'], authority)
        publication = signed(row['publication_job'], authority)
        manifest = signed(publication['manifest'], authority)
        cp = row['checkpoint']; files = cp['files']
        if (set(cp) != {'id', 'files'} or not isinstance(files, dict) or not 1 <= len(files) <= 32
                or 'config.json' not in files or not any(n.endswith('.safetensors') for n in files)
                or any(re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', n) is None
                       or re.fullmatch(r'[0-9a-f]{64}', h) is None for n, h in files.items())
                or file_map(files) != cp['id'] or cp['id'] in seen
                or cp['id'] in (ack['input_checkpoint']['id'], ack['new_checkpoint']['id'])):
            raise ValueError('safe distinct obsolete checkpoint inventory')
        seen.add(cp['id'])
        jobid = job.get('job_id', '')
        if (job.get('role') != 'train' or re.fullmatch(r'[A-Za-z0-9_-]{1,160}', jobid) is None
                or publication.get('role') != 'upload' or manifest['checkpoint']['id'] != cp['id']
                or manifest['checkpoint']['files'] != files):
            raise ValueError('original train and authenticated publication binding')
        path = root/'jobs'/jobid/'checkpoint-persistent-final'
        if str(path) != row['directory'] or path != path.resolve():
            raise ValueError('exact original owned export path')
        readback = row['readback']; objects = readback.get('objects', {})
        if (readback.get('kind') != 'independent-full-R2-export-readback-v1'
                or readback.get('checkpoint') != cp or set(objects) != set(files)
                or any(v.get('full_get') is not True or v.get('sha256') != files[n]
                       or type(v.get('size')) is not int or v['size'] <= 0
                       or v.get('key') != 'public/checkpoints/'+cp['id']+'/'+n
                       for n, v in objects.items())):
            raise ValueError('complete authenticated R2 byte readback required')
    return value, ack


def assert_unreferenced(paths, proc=Path('/proc')):
    roots = [str(Path(p)) for p in paths]
    for process in proc.iterdir():
        if not process.name.isdecimal(): continue
        try:
            state = (process/'stat').read_text().rsplit(')', 1)[1].split()[0]
            if state in ('Z', 'X'): continue
            for link in [process/'cwd', *list((process/'fd').iterdir())]:
                try: target = str(link.resolve(strict=True))
                except FileNotFoundError: continue
                if any(target == r or target.startswith(r+'/') for r in roots):
                    raise ValueError('obsolete export has live process reference')
            for line in (process/'maps').read_text().splitlines():
                fields = line.split(None, 5)
                if len(fields) == 6 and any(fields[5] == r or fields[5].startswith(r+'/') for r in roots):
                    raise ValueError('obsolete export has live memory mapping')
        except (FileNotFoundError, ProcessLookupError): continue


def current_guard(root, value, ack, authority):
    from subnet.backend_jobs import signed
    from subnet.trainer_cache_lifecycle import live_original
    marker = json.loads((root/'.cache-lifecycle/trainer-current-state.json').read_bytes())
    if marker.get('version') == 'reset-bound-trainer-retention-v1':
        if (set(marker) != {'version', 'ROOT_ack', 'lineage', 'retired_geneses', 'transition_sha256'}
                or signed(marker['ROOT_ack'], authority) != ack
                or marker['ROOT_ack'] != value['current_ack']):
            raise StaleCatalogError('reset retention marker current ACK changed')
        lineage = marker['lineage']
        if (set(lineage) != {'genesis_sha256', 'optimizer_steps', 'descriptor_sha256', 'inference_checkpoint'}
                or any(lineage[k] != ack['trainer_state'][k] for k in lineage)):
            raise StaleCatalogError('reset retention marker lineage changed')
    elif set(marker) == {'optimizer_steps', 'descriptor_sha256'}:
        lineage = marker
    else:
        raise ValueError('unsupported trainer retention marker format')
    if (lineage['optimizer_steps'] != ack['trainer_state']['optimizer_steps']
            or lineage['descriptor_sha256'] != ack['trainer_state']['descriptor_sha256']):
        raise StaleCatalogError('obsolete catalog is not anchored to current trainer state')
    cache = root/'.optimizer-state-cache'
    actual = json.loads((cache/'current.json').read_bytes())
    promotion = json.loads((cache/'promotion.json').read_bytes())
    if (actual.get('ROOT_ack') != value['current_ack'] or actual.get('job_id') != ack['job_id']
            or signed(actual['ROOT_ack'], authority) != ack
                or promotion.get('phase') != 'complete' or promotion.get('child_terminal_confirmed') is not True
                or promotion.get('ack') != value['current_ack'] or (cache/'pending.json').exists()):
        raise ValueError('current promotion must be terminal without pending optimizer state')
    for marker in (root/'runner-status').glob('*.json'):
        if live_original(json.loads(marker.read_bytes())):
            raise ValueError('trainer workspace has an original live process')


def retire_catalog(envelope, authority, workspace, *, apply=False):
    """Idempotent bounded discovery; eviction uses existing inode/lease guards.

    Apply is only called by a pinned operator after validating the signed
    catalog. No role process, cache mapping, historical report, or R2 object is
    altered. Current/pending optimizer state always survives.
    """
    from subnet.cache_lifecycle import CacheLifecycle, snapshot
    value, ack = validate_catalog(envelope, authority, workspace)
    root = Path(workspace).absolute()
    if all(not Path(row['directory']).exists() for row in value['candidates']):
        return dict(status='complete', removed_checkpoints=[], ready_checkpoints=[],
                    already_absent=[row['checkpoint']['id'] for row in value['candidates']],
                    reclaimable_bytes=0, optimizer_state_untouched=True,
                    current_checkpoint_not_read=True,
                    free_bytes=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize)
    lifecycle = CacheLifecycle(root)
    removed = []; ready = []; absent = []; total = 0
    with lifecycle.lease_checkpoint('trainer-state-retention', blocking=False):
        current_guard(root, value, ack, authority)
        for row in value['candidates']:
            path = Path(row['directory']); cp = row['checkpoint']; objects = row['readback']['objects']
            if not path.exists(): absent.append(cp['id']); continue
            with lifecycle.lease_checkpoint(cp['id'], blocking=False):
                if set(p.name for p in path.iterdir()) != set(cp['files']):
                    raise ValueError('obsolete export membership changed')
                assert_unreferenced([path]); before = {n: snapshot(path/n) for n in cp['files']}
                for name, digest in cp['files'].items():
                    h = hashlib.sha256()
                    with (path/name).open('rb') as stream:
                        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
                    if h.hexdigest() != digest or before[name]['size'] != objects[name]['size']:
                        raise ValueError('obsolete export bytes differ from durable inventory')
                after = {n: snapshot(path/n) for n in cp['files']}
                if before != after: raise ValueError('obsolete export changed during local hash')
                current_guard(root, value, ack, authority); assert_unreferenced([path])
                ready.append(cp['id']); total += sum(v['size'] for v in objects.values())
                if apply: lifecycle.adopt_checkpoint(cp['id'], path, cp['files'], envelope, expected_stats=after)
            if apply:
                current_guard(root, value, ack, authority); assert_unreferenced([path])
                actual = lifecycle.evict_checkpoints(exclude=(ack['new_checkpoint']['id'], ack['input_checkpoint']['id']), keep=0, only=(cp['id'],))
                if actual != [cp['id']]: raise ValueError('exact obsolete export eviction deferred')
                removed.extend(actual)
    return dict(status='complete' if apply else 'reviewed', removed_checkpoints=removed,
                ready_checkpoints=ready, already_absent=absent, reclaimable_bytes=total,
                optimizer_state_untouched=True, current_checkpoint=ack['new_checkpoint']['id'],
                free_bytes=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize)
