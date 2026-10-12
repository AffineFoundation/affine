"""Authenticate all fresh TRAIN records, full-GET archive, then publish aggregates."""
import base64
import hashlib
import json
from pathlib import Path
import time

from train_contract import canonical, digest, need, paired_comparability, signed, summarize, validate_plan
from serial_worker import save

sha = lambda raw: hashlib.sha256(raw).hexdigest()


def full_readback(bucket, key, raw):
    """Same bounded full-object verification as the qualified heldout archive."""
    need(0 < len(raw) <= 16 * 1024**2, 'bounded evaluation archive member')
    response = bucket.client.get_object(Bucket=bucket.name, Key=key)
    body = response['Body']
    try:
        need(response['ContentLength'] == len(raw) and body.read(len(raw) + 1) == raw,
             'durable archive exact size and full byte readback')
    finally:
        body.close()


def validate(envelope, files, authority):
    plan = signed(envelope, authority); validate_plan(plan, authority)
    result = json.loads(files['result.json'])
    names = {'task-' + str(t['index']) + '.json' for t in plan['training_cohort']['payload']['tasks']}
    need(set(files) == names | {'result.json'} and set(result['raw_artifacts']) == names,
         'complete original128 plus result; no selective exclusions')
    need(result['version'] == plan['version'] and result['plan_sha256'] == digest(envelope) and
         result['checkpoint'] == plan['checkpoint']['id'] and result['tasks'] == 128 and
         result['cohort_sha256'] == plan['cohort_sha256'] and result['profile_sha256'] == plan['profile_sha256'] and
         result['seeds_sha256'] == plan['seeds_sha256'], 'original result identity')
    rows = []
    for name in sorted(names):
        need(sha(files[name]) == result['raw_artifacts'][name], 'full raw task hash')
        rows.append(json.loads(files[name]))
    aggregate = summarize(plan, rows, exposure=plan.get('exposure'))
    need(result['correct'] == aggregate['solved'] and result['errors'] == aggregate['errors'] and
         result['unresolved'] == aggregate['unresolved'] and result['accuracy'] == aggregate['solved'] / 128,
         'raw/aggregate accounting equality')
    return plan, result, rows, aggregate


def capture_and_archive(monitor, job):
    code = 'DATA=' + repr(dict(root=monitor.remote, directory=monitor.remote + '/jobs/' + job)) + '\n' + '''from pathlib import Path
import base64,fcntl,json,os,stat
root=Path(DATA['root']);d=Path(DATA['directory']);fd=os.open(root/'serial.lease',os.O_RDONLY|os.O_NOFOLLOW);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
files={};total=0
for p in sorted((d/'output').iterdir()):
 s=p.lstat();assert stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_uid==os.getuid()and s.st_size<4*1024**2
 raw=p.read_bytes();a=p.stat();assert(s.st_ino,s.st_size,s.st_mtime_ns)==(a.st_ino,a.st_size,a.st_mtime_ns)
 total+=len(raw);assert total<64*1024**2;files[p.name]=base64.b64encode(raw).decode()
assert len(files)==129
print(json.dumps(dict(files=files)));os.close(fd)
'''
    files = {name: base64.b64decode(raw, validate=True) for name, raw in monitor.ssh_code(code)['files'].items()}
    local = monitor.root / 'jobs' / job
    envelope = json.loads((local / 'plan.json').read_bytes())
    plan, result, rows, aggregate = validate(envelope, files, monitor.authority)
    comparison = dict(comparable=True, denominator=128, failures=[])
    if plan['phase'] == 'latest':
        state = json.loads((monitor.root / 'state.json').read_bytes())
        baseline = monitor.root / 'jobs' / state['baseline_job']
        before_envelope = json.loads((baseline / 'plan.json').read_bytes())
        before_files = {p.name: p.read_bytes() for p in (baseline / 'captured').glob('task-*.json')}
        before_files['result.json'] = (baseline / 'captured/result.json').read_bytes()
        before_plan, _, before_rows, _ = validate(before_envelope, before_files, monitor.authority)
        comparison = paired_comparability(before_plan, before_rows, plan, rows)
    aggregate.update(generated_at=result['completed_at'], comparability_passed=comparison['comparable'])
    task_counts = plan.get('exposure_task_counts')
    private_metadata = [dict(t, training_task_exposures=None if task_counts is None else task_counts[str(t['index'])])
                        for t in plan['training_cohort']['payload']['tasks']]
    files.update({'plan.json': (local / 'plan.json').read_bytes(), 'evaluate.py': (local / 'evaluate.py').read_bytes(),
                  'train_contract.py': (local / 'train_contract.py').read_bytes(),
                  'comparability.json': canonical(comparison), 'task-exposure-metadata.json': canonical(private_metadata)})
    prefix = monitor.config['archive_prefix'] + '/' + digest(envelope) + '/'
    archive = {}; captured = local / 'captured'; captured.mkdir(mode=0o700, exist_ok=True)
    for name, raw in files.items():
        target = captured / name
        if target.exists(): need(target.read_bytes() == raw, 'immutable original captured bytes')
        else:
            with target.open('xb') as f: f.write(raw)
            target.chmod(0o600)
        key = prefix + name
        monitor.bucket.put(key, raw, 'application/octet-stream')
        monitor.full_readback(monitor.bucket, key, raw)
        archive[name] = dict(key=key, sha256=sha(raw), size=len(raw))
    aggregate['durable_result'] = dict(key=archive['result.json']['key'], sha256=archive['result.json']['sha256'], full_get_verified=True)
    # No aggregate becomes visible until EVERY raw task and dependency is durable.
    public = monitor.sign(aggregate); public_raw = canonical(public)
    public_key = prefix + 'aggregate.signed.json'
    monitor.bucket.put(public_key, public_raw, 'application/json'); monitor.full_readback(monitor.bucket, public_key, public_raw)
    receipt = monitor.sign(dict(version='prospective-training128-durable-readback-v1', at=time.time(),
                                checkpoint=plan['checkpoint']['id'], plan_sha256=digest(envelope),
                                result_sha256=archive['result.json']['sha256'], archive=archive,
                                aggregate_sha256=sha(public_raw), aggregate_key=public_key,
                                full_readback_verified=True, task_count=128, production_mutation=False,
                                optimizer_allocation=False, evaluation_lease_free_at_capture=True))
    receipt_path = local / 'archive.json'
    if receipt_path.exists():
        old = signed(json.loads(receipt_path.read_bytes()), monitor.authority)
        need(old['result_sha256'] == archive['result.json']['sha256'], 'same original durable result')
        receipt = json.loads(receipt_path.read_bytes())
    else:
        raw = canonical(receipt); key = prefix + 'ACK-' + sha(raw) + '.json'
        monitor.bucket.put(key, raw, 'application/json'); monitor.full_readback(monitor.bucket, key, raw)
        save(receipt_path, receipt)
    destination = Path(monitor.config['aggregate_directory'])
    destination.mkdir(parents=True, mode=0o700, exist_ok=True)
    target = destination / (digest({k:aggregate[k] for k in ('run_id','checkpoint','cohort_sha256','profile_sha256','seeds_sha256')}) + '.signed.json')
    if target.exists(): need(target.read_bytes() == public_raw, 'no conflicting aggregate overwrite')
    else: save(target, public)
    return receipt, aggregate
