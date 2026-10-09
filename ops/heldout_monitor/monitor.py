"""Serial, coalescing private held-out diagnostics; never gates training."""
import argparse
import base64
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from nacl.signing import SigningKey

from serial_worker import canonical, digest, file_hash, save, signed


def select_step(latest, completed, first=2, gap=2):
    """Coalesce to the latest durable checkpoint, rather than replaying backlog."""
    return latest >= first and (completed is None or latest >= completed + gap)


class Monitor:
    def __init__(self, config, authority):
        self.config = signed(json.loads(Path(config).read_bytes()), authority)
        self.authority = authority
        self.root = Path(self.config['local_root']); self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.config['version'] != 'serial-heldout128-monitor-v1': raise ValueError('scoped monitor config')
        for path, expected in self.config['pinned_files'].items():
            if file_hash(path) != expected: raise ValueError('pinned operational file')
        self.remote = self.config['remote_root']; self.ssh = self.config['ssh']
        sys.path.insert(0, self.config['study_root'])
        from archive import validate, normalized_summary, normalized_outcome, full_readback
        self.validate, self.summary, self.outcome, self.full_readback = validate, normalized_summary, normalized_outcome, full_readback
        self.template_envelope = json.loads(Path(self.config['base_plan']).read_bytes())
        self.template = signed(self.template_envelope, authority)
        self.base_archive = signed(json.loads(Path(self.config['base_archive']).read_bytes()), authority)
        if (self.base_archive['full_readback_verified'] is not True
                or self.base_archive['plan_sha256'] != digest(canonical(self.template_envelope))):
            raise ValueError('durable original baseline')
        raw = Path(self.config['authority_seed']).read_bytes().strip()
        self.key = SigningKey(raw if len(raw) == 32 else bytes.fromhex(raw.decode()))
        if self.key.verify_key.encode().hex() != authority: raise ValueError('same ROOT authority')
        sys.path.insert(0, self.config['bucket_source'])
        from subnet.storage import Bucket
        self.bucket = Bucket(json.loads(Path(self.config['bucket_config']).read_bytes())['bucket'])

    def sign(self, value):
        return dict(payload=value, signer=self.authority,
                    signature=base64.b64encode(self.key.sign(canonical(value)).signature).decode())

    def ssh_code(self, code, timeout=90):
        result = subprocess.run(self.ssh + ['python3 -B -'], input=code, capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            # Never print signed URLs, secret env, or arbitrary remote log contents.
            raise RuntimeError('remote monitor action failed with code ' + str(result.returncode))
        return json.loads(result.stdout)

    def stage(self, directory, files):
        rows = {name: base64.b64encode(raw).decode() for name, raw in files.items()}
        code = 'DATA=' + repr(dict(directory=directory, files=rows)) + '\n' + '''from pathlib import Path
import base64,json,os
os.umask(0o077);root=Path(DATA['directory']);assert root.is_absolute()and root.resolve()==root
root.mkdir(mode=0o700,parents=True,exist_ok=True)
for name,value in DATA['files'].items():
 assert Path(name).name==name
 raw=base64.b64decode(value,validate=True);path=root/name
 if path.exists():assert not path.is_symlink()and path.read_bytes()==raw
 else:
  with path.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  path.chmod(0o600)
print(json.dumps(dict(staged=True)))
'''
        return self.ssh_code(code)

    def worker(self, action, *, job=None, grant=None):
        args = [self.config['remote_python'], '-I', '-B', self.remote + '/serial_worker.py', action,
                '--root', self.remote, '--authority', self.authority, '--python', self.config['remote_python']]
        if job: args += ['--assignment', self.remote + '/jobs/' + job + '/assignment.json']
        if grant: args += ['--grant', grant]
        result = subprocess.run(self.ssh + [shlex.join(args)], capture_output=True, text=True, timeout=180)
        if result.returncode: raise RuntimeError('remote ' + action + ' failed with code ' + str(result.returncode))
        return json.loads(result.stdout)

    def latest(self):
        production = Path(self.config['production_state'])
        status = json.loads((production / 'controller.json').read_bytes())
        if status.get('persistent_state_committed') is not True: return None
        closure = status.get('last_completed_epoch', {})
        if not closure: return None
        original = signed(json.loads((production / (closure['epoch'] + '-signed-learner-completion.json')).read_bytes()), self.authority)
        pointer = json.loads((production / 'latest-trainer-state.json').read_bytes())
        if (closure != original or pointer != status['trainer_state']
                or closure['next_checkpoint'] != status['checkpoint']['id']
                or pointer['inference_checkpoint'] != closure['next_checkpoint']
                or pointer['optimizer_steps'] != status['public_optimizer_steps']):
            raise ValueError('genuine current durable training completion')
        if status['training_run_id'] != self.config['training_run_id'] or pointer['genesis_sha256'] != self.config['genesis_sha256']:
            raise ValueError('fresh run scope changed; preserve this study')
        return dict(checkpoint=status['checkpoint'], step=pointer['optimizer_steps'], closure=original, pointer=pointer)

    def prepare(self, latest, attempt=1, previous=None):
        cp = latest['checkpoint']; checkpoint = cp['id']; step = latest['step']
        key = 'public/checkpoints/' + checkpoint + '/authorities/' + self.authority + '/checkpoint.json'
        size = self.bucket.client.head_object(Bucket=self.bucket.name, Key=key)['ContentLength']
        if not 0 < size <= 128 * 1024: raise ValueError('bounded original model descriptor')
        descriptor = json.loads(self.bucket.get_bounded(key, limit=size)); actual = signed(descriptor, self.authority)
        if actual != {'id': checkpoint, 'files': cp['files']} or digest(canonical(actual['files'])) != checkpoint:
            raise ValueError('durable checkpoint exact full file map')
        job_id = 'step-' + str(step) + '-' + checkpoint[:16]
        if attempt > 1: job_id += '-attempt-' + str(attempt)
        local = self.root / 'jobs' / job_id; local.mkdir(mode=0o700, parents=True, exist_ok=True)
        if (local / 'assignment.json').exists():
            original = signed(json.loads((local / 'assignment.json').read_bytes()), self.authority)
            if any(file_hash(local / name) != value for name, value in original['program_files'].items()):
                raise ValueError('complete original prepared files required')
            return job_id
        prepared_path = local / 'prepared-files.json'
        if prepared_path.exists():
            prepared = json.loads(prepared_path.read_bytes())
            for name, body in prepared['files'].items():
                raw = base64.b64decode(body, validate=True); target = local / name
                if target.exists():
                    if target.read_bytes() != raw: raise ValueError('unchanged partially prepared job')
                else:
                    with target.open('xb') as stream:
                        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
                    target.chmod(0o600)
            save(local / 'production-completion.json', prepared['production_completion'])
            save(local / 'assignment.json', prepared['assignment'])
            return job_id
        directory = self.remote + '/jobs/' + job_id
        destination = str(Path(self.remote).parent / 'checkpoints' / checkpoint)
        objects, urls = {}, {}
        for name, checksum in actual['files'].items():
            member_key = 'public/checkpoints/' + checkpoint + '/' + name
            n = self.bucket.client.head_object(Bucket=self.bucket.name, Key=member_key)['ContentLength']
            objects[name] = dict(bytes=n, sha256=checksum)
            urls[name] = self.bucket.client.generate_presigned_url('get_object', Params={'Bucket': self.bucket.name, 'Key': member_key}, ExpiresIn=3600)
        now = time.time(); evaluate = copy.deepcopy(self.template)
        evaluate.update(directory=directory, created_at=now, expires_at=now + 7200,
                        checkpoint=dict(actual, path=destination, descriptor_key=key))
        read = dict(kind='immutable-checkpoint-read-hydration-v1', role='heldout128-evaluate', retained_UUID=self.config['retained_UUID'],
                    helper_sha256=file_hash(self.config['hydrate_program']), GPU_runs=0, optimizer_runs=0,
                    chain_transactions=0, publication_writes=0, created_at=now, expires_at=now + 3500,
                    checkpoint=checkpoint, checkpoint_authority=self.authority, checkpoint_descriptor=descriptor,
                    checkpoint_descriptor_sha256=digest(canonical(descriptor)), objects=objects, read_urls=urls,
                    destination=destination, allows_concurrent_scientific_reads=True)
        files = {'plan.json': canonical(self.sign(evaluate)), 'read-plan.json': canonical(self.sign(read)),
                 'evaluate.py': Path(self.config['evaluate_program']).read_bytes(), 'hydrate.py': Path(self.config['hydrate_program']).read_bytes()}
        body = dict(version='serial-heldout128-assignment-v1', root=self.remote, job_id=job_id,
                    checkpoint=checkpoint, checkpoint_path=destination, step=step,
                    attempt=attempt, retry_of=previous,
                    retained_UUID=self.config['retained_UUID'], worker_sha256=file_hash(self.config['worker_program']),
                    program_files={name: digest(raw) for name, raw in files.items()})
        envelope = self.sign(body)
        save(prepared_path, dict(files={name: base64.b64encode(raw).decode() for name, raw in files.items()},
                                 assignment=envelope, production_completion=latest))
        for name, raw in files.items():
            with (local / name).open('xb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            (local / name).chmod(0o600)
        save(local / 'production-completion.json', latest)
        save(local / 'assignment.json', envelope)
        return job_id

    def upload_job(self, job):
        local = self.root / 'jobs' / job
        files = {name: (local / name).read_bytes() for name in ('assignment.json', 'plan.json', 'read-plan.json', 'evaluate.py', 'hydrate.py')}
        self.stage(self.remote, {'serial_worker.py': Path(self.config['worker_program']).read_bytes()})
        self.stage(self.remote + '/jobs/' + job, files)

    def capture(self, job):
        code = 'DATA=' + repr(dict(root=self.remote, directory=self.remote + '/jobs/' + job)) + '\n' + '''import base64,fcntl,json,os,stat
from pathlib import Path
root=Path(DATA['root']);d=Path(DATA['directory']);fd=os.open(root/'serial.lease',os.O_RDONLY|os.O_NOFOLLOW);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert(d/'output/result.json').is_file();files={};total=0
for p in sorted((d/'output').iterdir()):
 s=p.lstat();assert stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_uid==os.getuid()and s.st_size<4*1024**2
 raw=p.read_bytes();after=p.stat();assert(s.st_ino,s.st_size,s.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 total+=len(raw);assert total<64*1024**2;files[p.name]=base64.b64encode(raw).decode()
assert len(files)==129
print(json.dumps(dict(files=files)));os.close(fd)
'''
        files = {name: base64.b64decode(raw, validate=True) for name, raw in self.ssh_code(code)['files'].items()}
        local = self.root / 'jobs' / job; plan_envelope = json.loads((local / 'plan.json').read_bytes())
        result, rows = self.validate(plan_envelope, files)
        base_files = {p.name: p.read_bytes() for p in Path(self.config['base_raw_directory']).glob('task-*.json')}
        base_files['result.json'] = (Path(self.config['base_raw_directory']) / 'result.json').read_bytes()
        _, baseline = self.validate(self.template_envelope, base_files)
        fixed = ('suites', 'cohort_sha256', 'source_files', 'runtime_versions', 'manifest', 'scientific_files', 'program_sha256')
        plan = signed(plan_envelope, self.authority)
        if any(plan[k] != self.template[k] for k in fixed): raise ValueError('frozen matched evaluation settings')
        before = {r['index']: r for r in baseline}; after = {r['index']: r for r in rows}
        if set(before) != set(after) or len(after) != 128: raise ValueError('all original128 tasks')
        same = lambda a, b: [t['output_tokens'] for t in a['raw_turns']] == [t['output_tokens'] for t in b['raw_turns']]
        analysis = dict(version='serial-matched-heldout128-result-v1', job_id=job, checkpoint=plan['checkpoint']['id'],
                        step=signed(json.loads((local / 'assignment.json').read_bytes()), self.authority)['step'],
                        baseline=self.summary(baseline), learned=self.summary(rows),
                        gains=sum(self.outcome(before[i]) != 'positive' and self.outcome(after[i]) == 'positive' for i in before),
                        losses=sum(self.outcome(before[i]) == 'positive' and self.outcome(after[i]) != 'positive' for i in before),
                        identical_output_sequences=sum(same(before[i], after[i]) for i in before),
                        fixed_seed_recipe='SHA256(affine-heldout128-draw-v1: + decimal(task_index))[:4], big-endian & 0x7fffffff',
                        same_per_task_draws=all(before[i]['seed'] == after[i]['seed'] for i in before),
                        same_generation_and_grading=True, proof_verification_performed=False,
                        raw_evidence_modified=False, production_barrier=False)
        if not analysis['same_per_task_draws']: raise ValueError('identical prescribed draws')
        files.update({'plan.json': (local / 'plan.json').read_bytes(), 'evaluate.py': (local / 'evaluate.py').read_bytes(),
                      'comparison.json': canonical(analysis)})
        receipt_path = local / 'archive.json'
        if receipt_path.exists():
            old = signed(json.loads(receipt_path.read_bytes()), self.authority)
            if old['result_sha256'] != digest(files['result.json']): raise ValueError('immutable captured result')
            return json.loads(receipt_path.read_bytes()), analysis
        prefix = self.config['archive_prefix'] + '/' + digest(canonical(plan_envelope)) + '/'
        archive = {}; raw_dir = local / 'captured'; raw_dir.mkdir(mode=0o700, exist_ok=True)
        for name, raw in files.items():
            target = raw_dir / name
            if target.exists():
                if target.read_bytes() != raw: raise ValueError('unchanged original output')
            else:
                with target.open('xb') as stream: stream.write(raw)
                target.chmod(0o600)
            key = prefix + name
            self.bucket.put(key, raw, 'text/plain' if name.endswith('.py') else 'application/json')
            self.full_readback(self.bucket, key, raw)
            archive[name] = dict(key=key, sha256=digest(raw), size=len(raw))
        receipt = self.sign(dict(version='prospective-heldout128-durable-readback-v1', at=time.time(),
                                 checkpoint=plan['checkpoint']['id'], plan_sha256=digest(canonical(plan_envelope)),
                                 result_sha256=digest(files['result.json']), archive=archive,
                                 full_readback_verified=True, task_count=128, derived_summary=self.summary(rows),
                                 production_mutation=False, optimizer_allocation=False, evaluation_lease_free_at_capture=True))
        raw = canonical(receipt); key = prefix + 'ACK-' + digest(raw) + '.json'
        self.bucket.put(key, raw, 'application/json'); self.full_readback(self.bucket, key, raw)
        save(receipt_path, receipt)
        return receipt, analysis

    def retirement(self, plan_envelope, archive, label):
        plan = signed(plan_envelope, self.authority)
        body = dict(version='archived-heldout128-model-retirement-v1', root=self.remote,
                    checkpoint=plan['checkpoint']['id'], evaluation_plan=plan_envelope, archive=archive)
        grant = self.sign(body); name = 'retire-' + label + '.json'
        self.stage(self.remote, {name: canonical(grant)})
        return self.worker('retire', grant=self.remote + '/' + name)

    def archive_failure(self, job):
        local = self.root / 'jobs' / job; receipt_path = local / 'failure-archive.json'
        if receipt_path.exists(): return json.loads(receipt_path.read_bytes())
        code = 'DATA=' + repr(dict(root=self.remote, directory=self.remote + '/jobs/' + job)) + '\n' + '''import base64,fcntl,json,os,stat
from pathlib import Path
root=Path(DATA['root']);d=Path(DATA['directory']);fd=os.open(root/'serial.lease',os.O_RDONLY|os.O_NOFOLLOW);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert(d/'dispatch.json').exists()and not(d/'output/result.json').exists()
status=json.loads((d/'status.json').read_bytes())if(d/'status.json').exists()else{}
phase='failed'if status.get('phase')=='failed'else'abandoned'
files={};total=0
paths=[p for p in d.iterdir()if p.is_file()and p.name not in('plan.json','read-plan.json','assignment.json','evaluate.py','hydrate.py','evaluation.lease')]
if(d/'output').exists():paths+=list((d/'output').iterdir())
for p in paths:
 s=p.lstat();assert stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_uid==os.getuid()and s.st_size<8*1024**2
 raw=p.read_bytes();after=p.stat();assert(s.st_ino,s.st_size,s.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 total+=len(raw);assert total<64*1024**2
 name=str(p.relative_to(d));files[name]=base64.b64encode(raw).decode()
print(json.dumps(dict(phase=phase,lease_free=True,files=files)));os.close(fd)
'''
        captured = self.ssh_code(code)
        files = {name: base64.b64decode(raw, validate=True) for name, raw in captured['files'].items()}
        for name in ('assignment.json', 'plan.json', 'read-plan.json'):
            files[name] = (local / name).read_bytes()
        assignment = signed(json.loads(files['assignment.json']), self.authority)
        archive = {}; prefix = self.config['archive_prefix'] + '/failed/' + digest(files['assignment.json']) + '/'
        for name, raw in files.items():
            if Path(name).is_absolute() or '..' in Path(name).parts: raise ValueError('bounded failed evidence path')
            key = prefix + name
            self.bucket.put(key, raw, 'application/octet-stream'); self.full_readback(self.bucket, key, raw)
            archive[name] = dict(key=key, sha256=digest(raw), size=len(raw))
        receipt = self.sign(dict(version='serial-heldout128-failed-attempt-v1', job_id=job,
                                 checkpoint=assignment['checkpoint'], assignment_sha256=digest(files['assignment.json']),
                                 read_plan_sha256=digest(files['read-plan.json']), phase=captured['phase'],
                                 full_readback_verified=True, scientific_success=False,
                                 archive=archive, at=time.time(), lease_free_at_capture=True))
        raw = canonical(receipt); key = prefix + 'ACK-' + digest(raw) + '.json'
        self.bucket.put(key, raw, 'application/json'); self.full_readback(self.bucket, key, raw)
        save(receipt_path, receipt)
        return receipt

    def recover_failed(self, state, job):
        local = self.root / 'jobs' / job
        assignment_envelope = json.loads((local / 'assignment.json').read_bytes())
        assignment = signed(assignment_envelope, self.authority)
        receipt = self.archive_failure(job)
        exhausted = assignment['attempt'] >= self.config['max_attempts']
        body = dict(version='failed-heldout128-cache-retirement-v1', root=self.remote,
                    assignment=assignment_envelope, read_plan=json.loads((local / 'read-plan.json').read_bytes()),
                    failure_archive=receipt, remove_complete_model=exhausted,
                    cleanup_helper_sha256=file_hash(self.config['failed_cleanup_program']))
        name = 'failed-retire-' + job + '.json'
        self.stage(self.remote, {name: canonical(self.sign(body))})
        cleanup = self.worker('retire-failed', grant=self.remote + '/' + name)
        save(local / 'failed-cache-retirement.json', cleanup)
        if exhausted:
            state.setdefault('failed', []).append(dict(job_id=job, checkpoint=assignment['checkpoint'],
                                                       step=assignment['step'], scientific_success=False))
            state.update(active=None, last_failed_step=assignment['step'])
            save(self.root / 'state.json', state)
            return
        latest = json.loads((local / 'production-completion.json').read_bytes())
        next_job = self.prepare(latest, attempt=assignment['attempt'] + 1,
                                previous=dict(job_id=job, failure_archive_sha256=digest(canonical(receipt))))
        state['active'] = next_job; save(self.root / 'state.json', state)
        self.upload_job(next_job); self.worker('dispatch', job=next_job)

    def tick(self):
        state_path = self.root / 'state.json'
        state = json.loads(state_path.read_bytes()) if state_path.exists() else dict(active=None, completed_step=None, completed=[])
        self.stage(self.remote, {'serial_worker.py': Path(self.config['worker_program']).read_bytes(),
                                'failed_cache_retirement.py': Path(self.config['failed_cleanup_program']).read_bytes()})
        if not state.get('legacy_retired'):
            legacy = self.config.get('completed_legacy_model')
            if legacy:
                receipt = self.retirement(json.loads(Path(legacy['plan']).read_bytes()), json.loads(Path(legacy['archive']).read_bytes()), 'prior-step1')
                save(self.root / 'legacy-retirement.json', receipt)
            state['legacy_retired'] = True; save(state_path, state)
        if state['active']:
            job = state['active']; self.upload_job(job)
            observed = self.worker('observe', job=job); save(self.root / 'observation.json', observed)
            if observed['phase'] == 'not-dispatched': self.worker('dispatch', job=job); return
            if observed['lease_busy']: return
            if observed['phase'] in ('failed', 'abandoned'):
                self.recover_failed(state, job); return
            if observed['phase'] in ('complete', 'complete-recovered') and observed['result_present']:
                archive, analysis = self.capture(job)
                local = self.root / 'jobs' / job
                receipt = self.retirement(json.loads((local / 'plan.json').read_bytes()), archive, job)
                save(local / 'retirement.json', receipt); save(local / 'comparison.json', analysis)
                state['completed'].append(dict(job_id=job, checkpoint=analysis['checkpoint'], step=analysis['step']))
                state.update(active=None, completed_step=analysis['step']); save(state_path, state)
                save(self.root / 'latest-comparison.json', analysis)
            return
        latest = self.latest()
        considered = [v for v in (state['completed_step'], state.get('last_failed_step')) if v is not None]
        previous_step = max(considered) if considered else None
        if latest and select_step(latest['step'], previous_step, self.config['first_step'], self.config['step_gap']):
            if any(x['checkpoint'] == latest['checkpoint']['id'] for x in state['completed']): return
            job = self.prepare(latest)
            state['active'] = job; save(state_path, state)
            self.upload_job(job); self.worker('dispatch', job=job)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    parser.add_argument('--authority', required=True); parser.add_argument('--once', action='store_true')
    args = parser.parse_args(); monitor = Monitor(args.config, args.authority)
    with (monitor.root / 'monitor.lease').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                monitor.tick()
                save(monitor.root / 'health.json', dict(at=time.time(), healthy=True, production_barrier=False))
            except Exception as error:
                save(monitor.root / 'health.json', dict(at=time.time(), healthy=False, error_type=type(error).__name__, production_barrier=False))
                if args.once: raise
            if args.once: break
            time.sleep(30)


if __name__ == '__main__': main()
