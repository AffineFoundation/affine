"""Explicit protected-replica lifecycle; never retire checkpoint identities.

Operator must authenticate full R2 readbacks, references and root approval before
invoking this node helper. This module never changes queue rows, protections or
worker configuration. The separately supervised worker may use retained keeper.
"""
import hashlib
import json
import os
import stat
import subprocess
import time
from pathlib import Path
from ops.verifier_cache_capacity_plan import plan_reuse


def private_save(path, value):
    with Path(path).open('x') as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, sort_keys=True)
        stream.flush(); os.fsync(stream.fileno())


def assert_unreferenced(roots, proc=Path('/proc')):
    roots = [str(Path(r)) for r in roots]
    for process in proc.iterdir():
        if not process.name.isdecimal(): continue
        try:
            fields = (process/'stat').read_text().rsplit(')',1)[1].split()
            if fields[0] in ('Z','X'): continue
            for obj in [process/'cwd', *list((process/'fd').iterdir())]:
                try: path = str(obj.resolve(strict=True))
                except FileNotFoundError: continue
                if any(path == r or path.startswith(r+'/') for r in roots):
                    raise ValueError('cache path has process reference')
            for line in (process/'maps').read_text().splitlines():
                values=line.split(None,5)
                if len(values)==6 and any(values[5]==r or values[5].startswith(r+'/') for r in roots):
                    raise ValueError('cache path is memory mapped')
        except (FileNotFoundError, ProcessLookupError): continue
        # Permission errors deliberately propagate: uncertain references refuse.


def retire_duplicate(plan, *, apply=False):
    """One-shot bounded retirement of an unreferenced physical duplicate.

    Protected keeper plus its exact second alias survive. Root approval is an
    operator-side signed-record check, never inferred from caller booleans here.
    A durable operation-start record is written BEFORE rename; any incomplete
    operation requires explicit forensic recovery rather than automatic rerun.
    """
    if (plan.get('kind') != 'explicit-verifier-redundancy-operation-v1'
            or plan.get('archive_readback_verified') is not True
            or plan.get('root_approval_verified') is not True
            or plan.get('queue_references_verified') is not True):
        raise ValueError('explicit authenticated operator approval required')
    original = plan['original_worker']
    process = Path('/proc')/str(original['pid'])
    if process.exists():
        fields=(process/'stat').read_text().rsplit(')',1)[1].split()
        if fields[19] == str(original['ticks']) and fields[0] not in ('Z','X'):
            raise ValueError('original worker must be terminal')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('idle GPU required')
    alias_root=Path(plan['keeper_alias'])
    if (not alias_root.is_absolute() or alias_root.resolve()!=alias_root
            or not stat.S_ISDIR(alias_root.lstat().st_mode)):
        raise ValueError('ordinary canonical keeper alias directory required')
    paths=[plan['keeper'],plan['keeper_alias'],plan['replica']]
    assert_unreferenced(paths)
    current=plan_reuse(plan['checkpoint'],plan['files'],plan['keeper'],plan['replica'],
                      plan['protected_checkpoints'],free_bytes=os.statvfs(plan['replica']).f_bavail*os.statvfs(plan['replica']).f_frsize,
                      artifact_bytes=plan['artifact_bytes'],reserve_bytes=plan['reserve_bytes'])
    if (not current['single_link_replacement_candidate'] or current['potential_reclaimed_bytes']<=0
            or not current['sufficient_capacity_after_prospective_reuse']):
        raise ValueError('single-link duplicate with adequate recoverable capacity required')
    for name,row in current['keeper_files'].items():
        alias=(Path(plan['keeper_alias'])/name).lstat()
        if (not stat.S_ISREG(alias.st_mode) or row['links']!=2 or alias.st_nlink!=2
                or (alias.st_dev,alias.st_ino)!=(row['device'],row['inode'])):
            raise ValueError('exact original keeper two-alias cluster required')
        expected=plan['prior_proof']['replica_files'][name]
        if current['replica_files'][name] != expected:
            raise ValueError('duplicate inode proof changed')
    if {p.name for p in Path(plan['keeper_alias']).iterdir()} != set(plan['files']):
        raise ValueError('keeper alias membership changed')
    # Refuse a newly started process after the expensive full-file readback.
    assert_unreferenced(paths)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPU state changed')
    result=dict(review_only=not apply,checkpoint=plan['checkpoint'],protected_checkpoints=plan['protected_checkpoints'],
                estimated_reclaim_bytes=current['potential_reclaimed_bytes'],keeper_paths_preserved=paths[:2],
                original_worker_terminal=True,free_bytes_before=current['free_bytes'])
    if not apply: return result
    journal=Path(plan['operation_directory'])
    if journal.exists(): raise ValueError('prior operation exists; observe without repeating')
    journal.mkdir(mode=0o700,parents=False)
    private_save(journal/'operation-start.private.json',dict(result,started_at=time.time(),plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()))
    root=Path(plan['replica']); retired=root.with_name(root.name+'.redundant-retired-'+journal.name)
    try:
        if retired.exists(): raise ValueError('retired directory exists')
        root.rename(retired)
        for name in plan['files']: (retired/name).unlink()
        retired.rmdir()
        result.update(completed=True,completed_at=time.time(),free_bytes_after=os.statvfs(root.parent).f_bavail*os.statvfs(root.parent).f_frsize,
                      replica_removed=True,global_protections_unchanged=True)
        private_save(journal/'operation-completed.private.json',result)
    except BaseException as error:
        private_save(journal/'operation-uncertain.private.json',dict(error_type=type(error).__name__,at=time.time(),automatic_repeat_forbidden=True))
        raise
    return result
