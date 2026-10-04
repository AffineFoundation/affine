"""Retire one idle, obsolete model cache using full authenticated R2 readback.

Scopes are explicit private operator records bound to the unchanged controller
config and endpoints. No recursive workspace discovery, GPU restart, job/export
deletion, or R2 mutation occurs here. Uncertain remote operations stop the watch.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from botocore.exceptions import ClientError
from subnet.storage import Bucket, canonical
from subnet.live_reward_bridge import signed
from subnet.remote_backend import RemoteJobs
from ops.retain_completed_training import guard, save, sha
from ops.retain_obsolete_training_exports import protected_checkpoints
from ops.retain_verifier_downloads import verified_archive


def endpoints(config):
    roles = config['remote']['roles']
    result = {name: roles[name] for name in ('train', 'evaluate', 'mine')}
    result.update({'verify' + str(i): e for i, e in enumerate(roles['verify'], 1)})
    return result


def private_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        raise ValueError('private operator file required')
    return path


def scopes_for(path, config_path, config):
    scopes = json.loads(private_file(path).read_text())
    peers = endpoints(config)
    if (scopes.get('schema') != 1 or scopes.get('config_sha256') != sha(config_path)
            or not isinstance(scopes.get('roles'), dict) or not scopes['roles']
            or set(scopes['roles']) - set(peers)):
        raise ValueError('explicit original config/cache scope binding')
    for name, row in scopes['roles'].items():
        roots = row.get('roots')
        if (row.get('endpoint_sha256') != hashlib.sha256(canonical(peers[name])).hexdigest()
                or not isinstance(roots, list) or not 1 <= len(roots) <= 16
                or len(set(roots)) != len(roots)):
            raise ValueError('bounded original endpoint/cache roots')
        for value in roots:
            if not isinstance(value, str):
                raise ValueError('explicit canonical cache root')
            root = Path(value)
            if (not root.is_absolute() or root.name not in ('checkpoint', 'checkpoints')
                    or len(root.parts) < 4
                    or not any(part.startswith('affine-') for part in root.parts[1:-1])
                    or '..' in root.parts or str(root) != value):
                raise ValueError('explicit canonical Affine cache root')
    return scopes['roles']


def protections(config_path, process_record, authority):
    config, principal = guard(config_path, process_record)
    _, all_protected = protected_checkpoints(config_path, process_record, authority)
    state = Path(config['state']); roles = state / 'roles'
    current = json.loads((state / 'controller.json').read_text())
    active = current.get('active') or {}
    if active.get('epoch'):
        manifest = signed(json.loads((state / (active['epoch'] + '-first-signed-manifest.json')).read_text()), authority)
        if manifest['epoch'] != active['epoch']:
            raise ValueError('original active manifest binding')
        all_protected.add(manifest['checkpoint']['id'])
    checker = RemoteJobs.__new__(RemoteJobs)
    checker.state = roles
    checker.controller = SimpleNamespace(authority=SimpleNamespace(id=authority))
    for path in roles.glob('*.json'):
        if path.name.endswith(('-job.json', '-report.json', '-failure.json')):
            continue
        prior = json.loads(path.read_text())
        if not isinstance(prior, dict) or prior.get('role') not in ('mine', 'train', 'evaluate'):
            continue
        # Labels include "before"/"after" and other operator choices. Select
        # the actual role field, never infer execution role from the filename.
        job = signed(json.loads((roles / (prior['job_id'] + '-job.json')).read_text()), authority)
        manifest = signed(job['manifest'], authority)
        if (job['role'] != prior['role'] or job['job_id'] != prior['job_id']
                or hashlib.sha256(canonical(job)).hexdigest() != prior['job_sha256']):
            raise ValueError('original dispatched job binding')
        report_path = roles / (prior['job_id'] + '-report.json')
        if report_path.exists():
            checker.checked(json.loads(report_path.read_text()), prior, manifest)
        else:
            # Neither expiry nor a failed observation establishes completion.
            all_protected.add(manifest['checkpoint']['id'])
    if (not 1 <= len(principal) <= 32 or len(all_protected) > 512
            or any(not isinstance(cp, str) or re.fullmatch('[0-9a-f]{64}', cp) is None
                   for cp in principal | all_protected)):
        raise ValueError('bounded current and active checkpoint protection')
    return config, principal, all_protected - principal


def archived_files(bucket, checkpoint, authority):
    key = 'public/checkpoints/' + checkpoint + '/authorities/' + authority + '/checkpoint.json'
    try:
        descriptor = signed(json.loads(bucket.get(key)), authority)
    except ClientError as error:
        if str(error.response.get('Error', {}).get('Code')) in ('NoSuchKey', '404', 'NotFound'):
            return None
        raise
    files = descriptor.get('files')
    if (descriptor.get('id') != checkpoint or not isinstance(files, dict)
            or not 1 <= len(files) <= 32 or 'config.json' not in files
            or not any(name.endswith('.safetensors') for name in files)
            or any(not isinstance(name, str) or re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_.-]*', name) is None
                   or Path(name).suffix not in {'.json', '.safetensors', '.txt', '.model', '.jinja', '.tiktoken'}
                   or not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None
                   for name, digest in files.items())
            or hashlib.sha256(canonical(files)).hexdigest() != checkpoint):
        raise ValueError('authenticated safe original checkpoint descriptor')
    def read(item):
        name, digest = item; object_key = 'public/checkpoints/' + checkpoint + '/' + name
        size = bucket.client.head_object(Bucket=bucket.name, Key=object_key)['ContentLength']
        if type(size) is not int or not 0 < size <= 5 * 1024 ** 3:
            raise ValueError('bounded archived checkpoint size')
        return name, verified_archive(bucket, dict(archive_key=object_key, sha256=digest, size=size))
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        return dict(pool.map(read, files.items()))


PROBE = '''import json,os,re,subprocess,time
from pathlib import Path
rows=[]
for value in ROOTS:
 base=Path(value)
 if not base.exists():continue
 if base.resolve()!=base or base.is_symlink():raise ValueError('canonical cache scope changed')
 members=list(base.iterdir())
 if len(members)>128:raise ValueError('bounded cache population')
 for p in members:
  if not re.fullmatch('[0-9a-f]{64}',p.name):continue
  if p.is_symlink() or not p.is_dir():raise ValueError('ordinary model cache required')
  files=list(p.iterdir())
  if not 1<=len(files)<=32:raise ValueError('bounded checkpoint membership')
  rows.append({'checkpoint':p.name,'directory':str(p)})
location=Path(ROOTS[0])
while not location.exists():location=location.parent
s=os.statvfs(location)
print(json.dumps({'at':time.time(),'candidates':rows,'free_bytes':s.f_bavail*s.f_frsize,'gpu_busy':bool(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip())}))
'''


def run_cycle(config_path, scopes_path, authority, output, process_record):
    os.umask(0o077); config_path = private_file(config_path)
    config, principal, active = protections(config_path, process_record, authority)
    scope_hash = sha(scopes_path); scopes = scopes_for(scopes_path, config_path, config)
    output = Path(output); output.mkdir(parents=True, mode=0o700, exist_ok=True)
    if output.is_symlink() or output.stat().st_mode & 0o077:
        raise ValueError('private checkpoint retention state')
    out = output / ('checkpoint-cache-' + str(time.time_ns())); out.mkdir(mode=0o700)
    peers = endpoints(config); transports = {}
    def remote(name, code, timeout=240):
        e = peers[name]; opts = ['-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ConnectTimeout=20', '-o', 'UserKnownHostsFile=' + e['known_hosts']]
        peer = e.get('user', 'root') + '@' + e['host']
        ssh = ['ssh', *opts, '-p', str(e['port']), peer]
        transports[name] = (ssh, ['scp', '-q', *opts, '-P', str(e['port'])], peer)
        result = subprocess.run(ssh + [shlex.quote(e['python']) + ' -I -B -c ' + shlex.quote(code)],
                                capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            save(out / ('transport-failure-' + str(time.time_ns()) + '.private.json'),
                 dict(role=name, at=time.time(), exit_code=result.returncode, stderr=result.stderr))
            raise RuntimeError('checkpoint housekeeping refused; retain original operation evidence')
        return json.loads(result.stdout)
    def observe(item):
        name, row = item
        actual = remote(name, 'ROOTS=' + repr(row['roots']) + '\n' + PROBE)
        save(out / (name + '-scoped-presence.private.json'), actual)
        return name, actual
    with concurrent.futures.ThreadPoolExecutor(min(4, len(scopes))) as pool:
        observations = dict(pool.map(observe, scopes.items()))
    save(out / 'original-scoped-presence.private.json', observations)
    candidates = [(name, row) for name, observation in observations.items() if not observation['gpu_busy']
                  for row in observation['candidates'] if row['checkpoint'] not in principal | active]
    candidates.sort(key=lambda item: (observations[item[0]]['free_bytes'], item[1]['directory']))
    bucket = None
    for name, selected in candidates:
        cp = selected['checkpoint']; bucket = bucket or Bucket(config['bucket'])
        readbacks = archived_files(bucket, cp, authority)
        if readbacks is None:
            continue  # No authenticated archive; preserve this replica unchanged.
        save(out / 'fresh-complete-public-readbacks.private.json', readbacks)
        _, principal, active = protections(config_path, process_record, authority)
        if cp in principal | active or sha(scopes_path) != scope_hash:
            raise ValueError('checkpoint scope/protection changed during archive verification')
        # A new GPU job during archive readback simply defers this cache.
        if remote(name, 'ROOTS=' + repr(scopes[name]['roots']) + '\n' + PROBE)['gpu_busy']:
            result = dict(removed_bytes=0, removed_replicas=0, deferred='role GPU occupied after archive readback')
            save(out / 'completion.private.json', result); return result
        plan = dict(checkpoint=cp, directory=selected['directory'], protected_checkpoints=sorted(principal),
                    active_checkpoints=sorted(active), archive_verified=True, descriptor_authenticated=True,
                    files={n: dict(sha256=v['sha256'], size=v['size']) for n, v in readbacks.items()})
        local = out / 'operator-plan.private.json'; save(local, plan)
        helper = Path(__file__).with_name('checkpoint_retention.py')
        e = peers[name]; location = str(Path(e['workspace']) / 'private-checkpoint-retention' / out.name)
        remote(name, 'from pathlib import Path;import json;Path(' + repr(location)
               + ').mkdir(parents=True,mode=0o700,exist_ok=False);print(json.dumps({"created":True}))')
        ssh, scp, peer = transports[name]
        for path, dest in [(helper, 'retention.py'), (local, 'plan.private.json')]:
            r = subprocess.run(scp + [str(path), peer + ':' + location + '/' + dest],
                               capture_output=True, text=True, timeout=180)
            if r.returncode:
                save(out / ('copy-failure-' + str(time.time_ns()) + '.private.json'), dict(exit_code=r.returncode, stderr=r.stderr))
                raise RuntimeError('checkpoint plan installation failed; no retirement issued')
        _, principal, active = protections(config_path, process_record, authority)
        if cp in principal | active or sha(scopes_path) != scope_hash:
            raise ValueError('checkpoint now referenced; retirement refused')
        operation = 'ROOT=' + repr(location) + '\nHELPER_SHA=' + repr(sha(helper)) + '\nPLAN_SHA=' + repr(sha(local)) + '\n' + '''import hashlib,importlib.util,json,os,time
from pathlib import Path
root=Path(ROOT);helper=root/'retention.py';p=root/'plan.private.json'
assert hashlib.sha256(helper.read_bytes()).hexdigest()==HELPER_SHA and hashlib.sha256(p.read_bytes()).hexdigest()==PLAN_SHA
spec=importlib.util.spec_from_file_location('retention',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
before=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize;result=m.remove_checkpoint_replica(json.loads(p.read_text()))
receipt={'at':time.time(),'result':result,'free_before':before,'free_after':os.statvfs(root).f_bavail*os.statvfs(root).f_frsize}
(root/'actual-completion.private.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
'''
        save(out / 'original-retirement.private.json', dict(role=name, remote=location,
             command_sha256=hashlib.sha256(operation.encode()).hexdigest(), helper_sha256=sha(helper), checkpoint=cp))
        actual = remote(name, operation, timeout=900); save(out / 'actual-retirement.private.json', actual)
        protections(config_path, process_record, authority)
        result = dict(removed_bytes=actual['result']['bytes'], removed_replicas=int(actual['result']['removed']),
                      role=name, checkpoint=cp, current_and_pending_preserved=True,
                      archive_objects_preserved=True, jobs_reports_exports_preserved=True)
        save(out / 'completion.private.json', result); return result
    result = dict(removed_bytes=0, removed_replicas=0, current_and_pending_preserved=True,
                  busy_roles=sum(row['gpu_busy'] for row in observations.values()))
    save(out / 'completion.private.json', result); return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'scopes', 'authority', 'output', 'controller-process'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--watch', action='store_true'); p.add_argument('--interval', type=int, default=300)
    a = p.parse_args()
    if not 60 <= a.interval <= 3600:
        raise ValueError('bounded retention interval')
    os.umask(0o077); output = Path(a.output); output.mkdir(parents=True, mode=0o700, exist_ok=True)
    if output.is_symlink() or output.stat().st_mode & 0o077:
        raise ValueError('private retention state')
    fd = os.open(output / 'checkpoint-cache.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            print(json.dumps(run_cycle(a.config, a.scopes, a.authority, output, a.controller_process)), flush=True)
            if not a.watch:
                return
            time.sleep(a.interval)


if __name__ == '__main__':
    main()
