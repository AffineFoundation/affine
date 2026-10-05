"""Authority-reviewed adoption of historical caches while leased workers run.

No filesystem discovery grants ownership. ROOT must sign fresh exact inode
snapshots and durable inventories. The original quiescent bootstrap contract is
deliberately not accepted here. All removals use the regular cache lifecycle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from subnet.cache_lifecycle import CacheLifecycle, identifier, snapshot
from subnet.distributed_roles import authenticate
from subnet.storage import canonical
from subnet.backend_jobs import file_map

VERSION = 'owned-verifier-cache-catalog-online-wrapper-v3'


def processes():
    result = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            arguments = (directory / 'cmdline').read_bytes().split(b'\0')
            result[int(directory.name)] = dict(
                start_ticks=int(fields[19]), parent=int(fields[1]),
                state=fields[0], arguments=[p.decode() for p in arguments if p])
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


def guard(value, observed=None):
    """Allow the pinned leased wrapper and its children, never legacy readers."""
    observed = processes() if observed is None else observed
    wrapper = value['expected_wrapper']
    pid = wrapper['pid']
    original = observed.get(pid)
    if not original or original['state'] == 'Z' or original['start_ticks'] != wrapper['start_ticks']:
        raise ValueError('original leased wrapper identity changed')
    files = wrapper['operator_files']
    if set(files) != {'wrapper', 'worker', 'lifecycle'}:
        raise ValueError('exact pinned leased operator files required')
    for record in files.values():
        path = Path(record['path'])
        if not path.is_absolute() or path != path.resolve():
            raise ValueError('operator path changed')
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('leased operator bytes changed')
    if files['wrapper']['path'] not in original['arguments']:
        raise ValueError('original wrapper invocation changed')
    if 'source_registry_sha256' in wrapper:
        arguments = original['arguments']
        if '--source-registry' not in arguments:
            raise ValueError('original source registry invocation required')
        registry = Path(arguments[arguments.index('--source-registry') + 1])
        if registry != registry.resolve() or hashlib.sha256(registry.read_bytes()).hexdigest() != wrapper['source_registry_sha256']:
            raise ValueError('original source registry changed')
    roots = [Path(entry['root']) for entry in value['roots']]
    for root in roots:
        if not root.is_absolute() or root != root.resolve() or not root.is_dir():
            raise ValueError('exact existing owned root required')
        arguments = original['arguments']
        if '--workspace' not in arguments or arguments[arguments.index('--workspace') + 1] != str(root.parent):
            raise ValueError('wrapper owned workspace binding')
    descendants = {pid}
    while True:
        added = {p for p, record in observed.items() if record['parent'] in descendants}
        if added <= descendants:
            break
        descendants |= added
    for process, record in observed.items():
        if process == os.getpid() or process in descendants or record['state'] == 'Z':
            continue
        # Legacy invocations referencing this workspace did not inherit leases.
        if any(str(root.parent) in argument for root in roots for argument in record['arguments']):
            raise ValueError('unleased workspace reader present')


def validate(value, now):
    if value.get('revision') != VERSION or value.get('managed_checkpoint_leases_required') is not True:
        raise ValueError('exact signed online ownership contract required')
    if not value['created_at'] <= now < value['expires_at'] or value['expires_at'] - value['created_at'] > 900:
        raise ValueError('fresh online ownership lifetime required')
    if not 1 <= len(value['roots']) <= 4:
        raise ValueError('bounded owned roots required')
    for entry in value['roots']:
        if not 0 <= len(entry['checkpoints']) <= 16 or len(entry.get('completed_downloads', [])) > 256:
            raise ValueError('bounded checkpoint catalog required')
        for checkpoint in entry['checkpoints']:
            cp = identifier(checkpoint['id'])
            ack = checkpoint['durability_ack']
            if file_map(checkpoint['files']) != cp or set(checkpoint['stats']) != set(checkpoint['files']):
                raise ValueError('checkpoint identity and exact observed member inventory')
            if ack.get('signed_manifest_verified') is not True or ack.get('independent_durable_hashes') is not True:
                raise ValueError('authenticated complete durable inventory required')


def retire_downloads(lifecycle, row, authority):
    """Retire an exact original job's inputs after ROOT-attested report ACK."""
    job = authenticate(row['original_signed_job'], authority)
    job_id = identifier(job['job_id'])
    manifest = authenticate(job['manifest'], authority)
    cp = manifest['checkpoint']['id']
    ack = row['coordinator_ack']
    if job.get('role') != 'verify' or ack.get('accepted') is not True:
        raise ValueError('original verification report ACK required')
    report_path = lifecycle.root / 'jobs' / job_id / 'report.json'
    report = json.loads(report_path.read_bytes())
    if (hashlib.sha256(canonical(job)).hexdigest() != ack['job_sha256'] or
            hashlib.sha256(canonical(report)).hexdigest() != ack['report_sha256'] or
            report.get('job_id') != job_id or report.get('job_sha256') != ack['job_sha256'] or
            report.get('success') is not True):
        raise ValueError('exact accepted original report binding')
    approved = {'submission-' + str(i) + '.zip': obj['sha256'] for i, obj in enumerate(job['submissions'])}
    if set(row['files']) - set(approved):
        raise ValueError('original verified submission inventory')
    with lifecycle.lease_checkpoint(cp, blocking=False):
        for name, record in row['files'].items():
            path = lifecycle.root / 'jobs' / job_id / name
            if record['sha256'] != approved[name] or snapshot(path) != record['stat']:
                raise ValueError('accepted original submission inode changed')
        for name, record in row['files'].items():
            lifecycle.record_download(lifecycle.root / 'jobs' / job_id / name, record['sha256'])
        return lifecycle.retire_downloads(job_id, only={str(Path('jobs') / job_id / name) for name in row['files']})


def apply(envelope, authority, *, now=None, observed=None):
    value = authenticate(envelope, authority)
    validate(value, time.time() if now is None else now)
    guard(value, observed)
    results = []
    for entry in value['roots']:
        root = Path(entry['root'])
        lifecycle = CacheLifecycle(root)
        adopted, skipped = [], []
        for checkpoint in entry['checkpoints']:
            cp = checkpoint['id']
            try:
                with lifecycle.lease_checkpoint(cp, blocking=False):
                    directory = root / 'checkpoints' / cp
                    if not directory.exists():
                        skipped.append(dict(checkpoint=cp, reason='absent'))
                        continue
                    if set(p.name for p in directory.iterdir()) != set(checkpoint['files']):
                        raise ValueError('historical cache inventory changed')
                    if any(snapshot(directory / name) != record for name, record in checkpoint['stats'].items()):
                        raise ValueError('historical cache inode changed')
                    lifecycle.adopt_checkpoint(cp, directory, checkpoint['files'], checkpoint['durability_ack'], expected_stats=checkpoint['stats'])
                    adopted.append(cp)
            except BlockingIOError:
                skipped.append(dict(checkpoint=cp, reason='leased'))
            except (ValueError, FileNotFoundError):
                skipped.append(dict(checkpoint=cp, reason='changed-or-unowned'))
        # An online worker may have hydrated a newer checkpoint since this
        # catalog was observed. Never evict a newly receipted, unlisted model.
        removed = lifecycle.evict_checkpoints(exclude=entry['keep'], keep=0, only=set(adopted))
        downloads = []
        for row in entry.get('completed_downloads', []):
            try:
                downloads.extend(retire_downloads(lifecycle, row, authority))
            except (BlockingIOError, ValueError, FileNotFoundError):
                skipped.append(dict(job_id=row['original_signed_job']['payload']['job_id'], reason='input-busy-changed-or-unacknowledged'))
        results.append(dict(root=str(root), adopted=adopted, removed=removed, skipped=skipped,
                            retired_downloads=downloads,
                            available_bytes=__import__('shutil').disk_usage(root).free))
    return results


def prepare(template, operator_overlay):
    """Return unsigned fresh observations only; never create receipts/delete."""
    value = json.loads(json.dumps(template))
    if value.get('revision') != 'owned-verifier-cache-catalog-online-wrapper-v2':
        raise ValueError('explicit previously reviewed preparation template required')
    overlay = Path(operator_overlay)
    value['revision'] = VERSION
    value['created_at'] = time.time()
    value['expires_at'] = value['created_at'] + 900
    value['expected_wrapper']['operator_files'] = {
        name: dict(path=str(overlay / relative), sha256=hashlib.sha256((overlay / relative).read_bytes()).hexdigest())
        for name, relative in [('wrapper', 'ops/automatic_verifier_lifecycle_worker.py'),
                               ('worker', 'subnet/distributed_worker.py'),
                               ('lifecycle', 'subnet/cache_lifecycle.py')]}
    guard(value)
    for entry in value['roots']:
        root = Path(entry['root'])
        for checkpoint in entry['checkpoints']:
            directory = root / 'checkpoints' / checkpoint['id']
            checkpoint['stats'] = {name: snapshot(directory / name) for name in checkpoint['files']}
    validate(value, value['created_at'])
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--authority')
    parser.add_argument('--prepare-with-operator-overlay')
    args = parser.parse_args()
    value = json.loads(Path(args.catalog).read_text())
    if args.prepare_with_operator_overlay:
        result = prepare(value, args.prepare_with_operator_overlay)
    else:
        if not args.authority:
            parser.error('--authority is required for applying a signed catalog')
        result = apply(value, args.authority)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
