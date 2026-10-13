"""Independent checkpoint evaluation on the existing authenticated role queue.

The baseline is evaluated first. If training outruns evaluation, subsequently
select the newest checkpoint and record which intermediate points were skipped.
Never hold up an epoch or turn a failed evaluator into a reward of zero.
"""
import fcntl
import json
import os
from pathlib import Path
import time

from .backend_jobs import signed
from .controller import save_manifest
from .mimo_release_policy import digest, need, sign
from .mimo_evaluation import VERSION as EVALUATION, validate_sets
from .mimo_worker import VERSION as WORKER, validate, validate_report
from .complete_group_capture import _immutable
from .storage import canonical


class EvaluationScheduler:
    def __init__(self, directory, bucket, queue, identity, *, policy, metadata,
                 workers, set_name='fixed32', job_seconds=14400, max_retries=2, source_registry=None):
        self.root = Path(directory).absolute()
        need(self.root.resolve() == self.root, 'ordinary evaluation state')
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        need(self.root.stat().st_uid == os.getuid() and not self.root.stat().st_mode & 0o077,
             'owner-only evaluation state')
        need(type(job_seconds) is int and 60 <= job_seconds <= 86400
             and type(max_retries) is int and 0 <= max_retries <= 8,
             'bounded evaluation lifetime and retries')
        need(queue.authority == identity.id == policy.authority and callable(metadata),
             'same authenticated evaluator authority')
        need(type(workers) is list and workers and len(set(workers)) == len(workers)
             and all('evaluate' in queue.workers.get(w, []) for w in workers),
             'explicit admitted evaluator worker roster')
        self.bucket, self.queue, self.identity = bucket, queue, identity
        self.policy, self.metadata, self.workers = policy, metadata, set(workers)
        self.source_registry = source_registry
        if source_registry is not None:
            from .mimo_source_migration import SourceReleaseRegistry
            need(isinstance(source_registry, SourceReleaseRegistry)
                 and source_registry.authority == identity.id
                 and source_registry.active.policy_sha256 == policy.policy_sha256,
                 'explicit active evaluation release registry')
        self.set_name, self.seconds, self.retries = set_name, job_seconds, max_retries

    def _policy_for(self, manifest):
        if self.source_registry is not None:
            return self.source_registry.resolve(manifest)
        self.policy.validate_manifest(manifest)
        return self.policy

    def _state(self):
        path = self.root/'state.json'
        if not path.exists():
            return dict(version='mimo-evaluation-scheduler-v1', policy_sha256=self.policy.policy_sha256,
                        set_name=self.set_name, checkpoints={})
        need(not path.is_symlink() and path.stat().st_size <= 32*1024**2,
             'bounded original evaluator state')
        state = json.loads(path.read_bytes())
        need(state['version'] == 'mimo-evaluation-scheduler-v1'
             and (state['policy_sha256'] == self.policy.policy_sha256 or
                  self.source_registry is not None and state['policy_sha256'] in self.source_registry.policies)
             and state['set_name'] == self.set_name, 'same evaluator release and suite')
        return state

    def tick(self, openings, *, now=None):
        now = time.time() if now is None else now
        fd = os.open(self.root/'evaluation.lock', os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
            state = self._state()
            for opening in openings:
                m = signed(opening, self.identity.id); self._policy_for(m)
                suites = validate_sets(m)
                need(self.set_name in suites, 'original signed evaluation suite required')
                key = digest([m['checkpoint']['id'], suites[self.set_name]['dataset_id']])
                if key not in state['checkpoints']:
                    path = self.root/(key+'.opening.json')
                    save_manifest(path, opening)
                    state['checkpoints'][key] = dict(opening_sha256=digest(opening),
                        checkpoint=m['checkpoint']['id'], first_seen=m['start'], generation=0,
                        status='pending', job_id=None)
            save_manifest(self.root/'state.json', state)
            active = []
            for key, row in state['checkpoints'].items():
                if not row['job_id'] or row['status'] in ('complete', 'error', 'exhausted'): continue
                with self.queue.transaction() as db:
                    q = db.execute('SELECT * FROM jobs WHERE id=?', (row['job_id'],)).fetchone()
                    q = None if q is None else dict(q)
                path = self.root/(row['job_id']+'.job.json')
                envelope = json.loads(path.read_bytes()); job = signed(envelope, self.identity.id)
                if q is not None and q['status'] == 'complete':
                    response = json.loads(q['report_request'])
                    need(response.get('signer') in self.workers, 'admitted evaluator report signer')
                    request = signed(response, response['signer'])
                    need(request.get('action') == 'report' and request.get('job_id') == row['job_id'],
                         'original evaluator queue report')
                    report = request['report']; m = signed(job['manifest'], self.identity.id)
                    original_policy = self._policy_for(m)
                    if self.source_registry is not None:
                        self.source_registry.validate_job_execution(job)
                    validate_report(report, job, m, self.identity.id, digest(job), now)
                    receipt = sign(self.identity, dict(version='mimo-evaluation-receipt-v1',
                        policy_sha256=original_policy.policy_sha256, job=envelope, worker_report=response))
                    _immutable(self.bucket, original_policy.namespace()+'/evaluations/'+row['job_id']+'.json',
                               canonical(receipt))
                    save_manifest(self.root/(row['job_id']+'.receipt.json'), receipt)
                    row['receipt_sha256'] = digest(receipt)
                    if report['evaluation']['status'] == 'complete':
                        row['status'] = 'complete'
                    else:
                        # Preserve the original error receipt, then retry the
                        # same signed suite/checkpoint within the stated cap.
                        row.setdefault('prior_jobs', []).append(row['job_id'])
                        row['job_id'] = None
                        if row['generation'] >= self.retries: row['status'] = 'exhausted'
                        else: row.update(generation=row['generation']+1, status='pending')
                    continue
                if job['expires_at'] <= now or q is not None and q['status'] in ('failed', 'expired'):
                    row.setdefault('prior_jobs', []).append(row['job_id'])
                    row['job_id'] = None
                    if row['generation'] >= self.retries: row['status'] = 'exhausted'
                    else: row.update(generation=row['generation']+1, status='pending')
                    continue
                if q is None: self.queue.enqueue(envelope)  # Recover original enqueue lost ACK.
                active.append(row['job_id'])
            save_manifest(self.root/'state.json', state)
            if active: return dict(status='evaluating', jobs=active)
            pending = [(k, r) for k, r in state['checkpoints'].items() if r['status'] in ('pending', 'skipped')]
            if not pending: return dict(status='caught-up')
            baseline = self.policy.scope['initial_checkpoint']['id']
            first = [pair for pair in pending if pair[1]['checkpoint'] == baseline]
            key, row = min(first, key=lambda p:p[1]['first_seen']) if first else max(pending, key=lambda p:p[1]['first_seen'])
            for other_key, other in pending:
                if other_key != key and other['first_seen'] < row['first_seen']: other['status'] = 'skipped'
            opening = json.loads((self.root/(key+'.opening.json')).read_bytes())
            need(digest(opening) == row['opening_sha256'], 'original checkpoint opening bytes')
            m = signed(opening, self.identity.id); self._policy_for(m)
            identifier = 'mimo-eval-'+key[:40]+'-'+str(row['generation'])
            path = self.root/(identifier+'.job.json')
            if path.exists():
                envelope = json.loads(path.read_bytes()); job = signed(envelope, self.identity.id)
                need(job['job_id'] == identifier and job['manifest'] == opening,
                     'original evaluation job after interrupted state write')
            else:
                metadata = dict(self.metadata(m))
                if self.source_registry is not None:
                    metadata.update(self.source_registry.execution_for(m))
                job = dict(schema=1, worker_policy=WORKER, job_id=identifier, role='evaluate',
                    created_at=now, expires_at=now+self.seconds, manifest=opening,
                    evaluation=dict(version=EVALUATION, run_id=m['run_id'], set_name=self.set_name),
                    **metadata)
                envelope = sign(self.identity, job)
                validate(envelope, self.identity.id, now, allow_public=self.policy.mode == 'public')
                save_manifest(path, envelope)
            row.update(job_id=identifier, status='queued')
            save_manifest(self.root/'state.json', state)
            if job['expires_at'] > now: self.queue.enqueue(envelope)
            return dict(status='enqueued', job_id=identifier, checkpoint=row['checkpoint'])
        finally:
            os.close(fd)
