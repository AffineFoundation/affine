"""Bounded read-only retrieval of original logs for already finalized epochs.

Only the local private cache is written. Missing logs and malformed evidence are
reported per epoch and cannot stop the separate finalized-evidence publisher.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from nacl.exceptions import BadSignatureError
from dashboard import training_evidence_projection as projection

MAX_LOG_BYTES = 1024**2
MAX_TOTAL_BYTES = 8*1024**2
REMOTE_CODE = r'''
from pathlib import Path
import base64,hashlib,json,time
root=Path(DATA['workspace']);results=[];total=0
for row in DATA['rows']:
    result=dict(row,retrieved_at=time.time())
    try:
        job=root/(row['job_id']+'.json');log=root/(row['job_id']+'-worker.log')
        if not log.exists():
            result['status']='not_retained_on_current_trainer'
        elif (job.is_symlink() or not job.is_file() or job.stat().st_size>32*1024**2
              or hashlib.sha256(job.read_bytes()).hexdigest()!=row['job_raw_sha256']):
            result['status']='original_job_binding_unavailable'
        elif log.is_symlink() or not log.is_file() or log.stat().st_size>1048576:
            result['status']='log_not_bounded_regular_file'
        else:
            raw=log.read_bytes();total+=len(raw)
            if len(raw)>1048576 or total>8*1024**2:
                result['status']='retrieval_byte_budget_exhausted'
            else:
                result.update(status='available',raw_sha256=hashlib.sha256(raw).hexdigest(),
                    raw_size=len(raw),raw_base64=base64.b64encode(raw).decode(),original_job_bytes_match=True)
    except OSError:
        result['status']='original_log_read_error'
    results.append(result)
print(json.dumps(results))
'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def immutable_file(path, raw):
    import tempfile
    def existing_matches():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise ValueError('immutable_log_cache_collision')
    if path.exists() or path.is_symlink():
        existing_matches()
        return
    descriptor, name = tempfile.mkstemp(prefix='.'+path.name+'.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        try:
            # A reader sees the complete fsynced inode; another writer's final
            # name is never replaced, even if it appeared after the first check.
            os.link(temporary, path)
        except FileExistsError:
            existing_matches()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def requests(state, training_cache, directory, authority, now, max_jobs):
    rows = []
    skipped = []
    for completion_path in sorted(state.glob('*-signed-learner-completion.json')):
        epoch = completion_path.name.removesuffix('-signed-learner-completion.json')
        if not projection.EPOCH.fullmatch(epoch) or int(epoch.rsplit('-', 1)[1]) < 14:
            continue
        try:
            manifest_path = state/(epoch+'-first-signed-manifest.json')
            manifest, completion = projection.finalized(epoch, {
                'manifest_envelope': projection.read_json(manifest_path),
                'completion_envelope': projection.read_json(completion_path)}, authority)
            metrics = projection.authenticated(projection.read_json(training_cache/(epoch+'.signed.json')), authority)
            if set(metrics) == {'checkpoint', 'status'} and metrics['status'] == 'closed_no_eligible_batches':
                if metrics['checkpoint'] != completion['checkpoint'] or completion['checkpoint'] != completion['next_checkpoint']:
                    raise ValueError('empty_epoch_checkpoint')
                continue
            if (metrics['source_epoch'] != epoch or metrics['input_checkpoint'] != completion['checkpoint']
                    or metrics['checkpoint'] != completion['next_checkpoint'] or metrics['state_authority_committed'] is not True):
                raise ValueError('completed_training_receipt_binding')
            job_id = metrics['remote_job_id']
            if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,199}', job_id):
                raise ValueError('safe_original_job_identifier')
            job_path = state/'roles'/(job_id+'-job.json')
            job_document = projection.read_json(job_path)
            job = projection.authenticated(job_document, authority)
            admitted = projection.authenticated(job['manifest'], authority)
            if (job['role'] != 'train' or job['job_id'] != job_id or admitted['epoch'] != epoch
                    or admitted['checkpoint']['id'] != manifest['checkpoint']['id']
                    or projection.digest(job) != metrics['original_job_sha256']):
                raise ValueError('original_finalized_job_binding')
            row = {'epoch': epoch, 'round': completion['round'], 'job_id': job_id,
                   'job_raw_sha256': sha(job_path.read_bytes())}
            log_path = directory/(epoch+'.worker.log')
            receipt_path = directory/(epoch+'.receipt.json')
            if log_path.exists() and receipt_path.exists():
                raw = log_path.read_bytes()
                if len(raw) > MAX_LOG_BYTES:
                    raise ValueError('bounded_cached_log')
                projection.trainer_log_projection({'raw': raw, 'receipt': projection.read_json(receipt_path)},
                                                  epoch, job, row['job_raw_sha256'])
                continue
            retry = directory/(epoch+'.availability.json')
            if retry.is_file() and not retry.is_symlink():
                prior = projection.read_json(retry)
                if prior.get('job_raw_sha256') == row['job_raw_sha256'] and 0 <= now-prior.get('checked_at', 0) < 6*3600:
                    continue
            if len(rows) < max_jobs:
                rows.append(row)
        except (KeyError, ValueError, OSError, TypeError, BadSignatureError):
            skipped.append({'epoch': epoch, 'status': 'finalized_log_source_unavailable'})
    return rows, skipped


def _collect(state, training_cache, directory, remote, authority=projection.AUTHORITY, *, max_jobs=16, run=subprocess.run):
    import base64
    from nacl.exceptions import BadSignatureError
    if type(max_jobs) is not int or not 1 <= max_jobs <= 32:
        raise ValueError('bounded_log_job_count')
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError('owned_private_log_cache')
    try:
        rows, skipped = requests(Path(state), Path(training_cache), directory, authority, time.time(), max_jobs)
    except BadSignatureError:
        return {'retrieved': 0, 'status': 'source_signature_error'}
    if not rows:
        return {'retrieved': 0, 'skipped': skipped, 'status': 'no_uncached_finalized_logs'}
    data = {'rows': rows, 'workspace': remote['workspace']}
    code = 'DATA = '+repr(data)+'\n'+REMOTE_CODE
    argv = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10',
            '-o', 'UserKnownHostsFile='+remote['known_hosts'], '-p', str(remote['port']),
            remote.get('user', 'root')+'@'+remote['host'], shlex.quote(remote['python'])+' -I -B -']
    try:
        result = run(argv, input=code, text=True, capture_output=True, timeout=40)
        if result.returncode or len(result.stdout) > 12*1024**2:
            raise ValueError('bounded_remote_log_response')
        responses = json.loads(result.stdout)
        if type(responses) is not list or len(responses) != len(rows):
            raise ValueError('exact_log_response_inventory')
    except (OSError, ValueError, subprocess.SubprocessError):
        return {'retrieved': 0, 'skipped': skipped, 'status': 'remote_log_read_unavailable'}
    retrieved = 0
    missing = []
    total = 0
    for expected, response in zip(rows, responses):
        try:
            if any(response.get(key) != value for key, value in expected.items()):
                raise ValueError('exact_original_log_response')
            if response.get('status') != 'available':
                missing.append({'epoch': expected['epoch'], 'status': 'original_log_unavailable'})
                retry = dict(expected, checked_at=time.time(), status='original_log_unavailable')
                path = directory/(expected['epoch']+'.availability.json')
                temporary = path.with_suffix('.tmp')
                temporary.write_bytes(projection.canonical(retry))
                temporary.chmod(0o600)
                os.replace(temporary, path)
                continue
            raw = base64.b64decode(response.pop('raw_base64'), validate=True)
            total += len(raw)
            if (len(raw) > MAX_LOG_BYTES or total > MAX_TOTAL_BYTES or sha(raw) != response['raw_sha256']
                    or len(raw) != response['raw_size'] or response.get('original_job_bytes_match') is not True):
                raise ValueError('bounded_original_log_bytes')
            immutable_file(directory/(expected['epoch']+'.worker.log'), raw)
            immutable_file(directory/(expected['epoch']+'.receipt.json'), projection.canonical(response))
            retrieved += 1
        except (KeyError, TypeError, ValueError, OSError):
            missing.append({'epoch': expected['epoch'], 'status': 'log_cache_binding_error'})
    return {'retrieved': retrieved, 'bytes': total, 'missing': missing, 'skipped': skipped, 'status': 'complete'}



def collect(state, training_cache, directory, remote, authority=projection.AUTHORITY, *, max_jobs=16, run=subprocess.run):
    import fcntl
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError('owned_private_log_cache')
    descriptor = os.open(directory/'collector.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'another_log_collector_active', 'retrieved': 0}
        return _collect(state, training_cache, directory, remote, authority, max_jobs=max_jobs, run=run)
    finally:
        os.close(descriptor)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--training-cache', type=Path, required=True)
    parser.add_argument('--max-jobs', type=int, default=16)
    args = parser.parse_args()
    from ops.publish_live_public_discovery import actual_selector
    config, _ = actual_selector(projection.AUTHORITY)
    result = collect(config['state'], args.training_cache, args.directory,
                     config['remote']['roles']['train'], max_jobs=args.max_jobs)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
