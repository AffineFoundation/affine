"""Miner role: decrypt capability, pin checkpoint, search and upload cumulative batches."""
import requests
import time
from pathlib import Path
from .runtime_factory import runtime as make_runtime
from .model import check_runtime_profile
from .batches import pack,unpack
from .artifact_budget import for_manifest
from .protocol import entry,entries,harness_for, classification
from .storage import canonical
import hashlib
from verifiers.v1.errors import TaskError

class EpochClosed(RuntimeError):
    """The original signed upload window has ended; not an invalid sample."""

class MinerProgress:
    """Bounded private metadata only; observation failures never affect mining."""
    def __init__(self, path, manifest):
        self.path = Path(path)
        self.value = dict(schema=1, epoch=manifest['epoch'], checkpoint=manifest['checkpoint']['id'],
            source_sha256=manifest.get('source_bundle', {}).get('sha256'), started_at=time.time(),
            cumulative_generation_elapsed_seconds=0.0,
            counts={k: 0 for k in ('tasks_started','tasks_ended','attempts_started','attempts_ended',
                'completed_rollouts','positive','negative','neutral','indeterminate','error','deadline')})

    def record(self, event, *, env_id=None, index=None, attempt=None,
               outcome=None, elapsed=None, batches=0, error_type=None):
        import json, os, math
        try:
            value = self.value
            if event == 'task_start':
                value['counts']['tasks_started'] += 1
                self.task_started = time.monotonic()
                value['task_started_at'] = time.time()
            if event in ('batch_complete','task_exhausted','task_deadline','task_error'):
                value['counts']['tasks_ended'] += 1
                value['task_elapsed_seconds'] = max(0.0, time.monotonic()-getattr(self, 'task_started', time.monotonic()))
            if event in ('task_deadline','epoch_deadline') and value.get('last_attempt',{}).get('outcome')!='deadline': value['counts']['deadline'] += 1
            if event == 'attempt_start': value['counts']['attempts_started'] += 1
            if event == 'attempt_end':
                duration = float(elapsed) if elapsed is not None else 0.0
                if not math.isfinite(duration) or duration < 0: duration = 0.0
                total = value['cumulative_generation_elapsed_seconds'] + duration
                if math.isfinite(total): value['cumulative_generation_elapsed_seconds'] = total
                value['counts']['attempts_ended'] += 1
                if outcome in ('positive','negative','neutral'):
                    value['counts']['completed_rollouts'] += 1
                if outcome in ('positive','negative','neutral','indeterminate','error','deadline'):
                    value['counts'][outcome] += 1
            value['event'] = event
            value['observed_at'] = time.time()
            value['completed_batch_count'] = batches
            if env_id is not None: value['last_task'] = dict(env_id=env_id, index=index)
            if attempt is not None:
                previous = value.get('last_attempt', {})
                started_at = time.time() if event == 'attempt_start' else previous.get('started_at')
                value['last_attempt'] = dict(attempt=attempt, outcome=outcome,
                    started_at=started_at, ended_at=time.time() if event=='attempt_end' else None,
                    elapsed_seconds=duration if event=="attempt_end" else None, error_type=error_type)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix('.tmp-'+str(os.getpid()))
            descriptor = os.open(temporary, os.O_WRONLY|os.O_CREAT|os.O_TRUNC|os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, 'w') as stream:
                os.chmod(temporary, 0o600)
                json.dump(value, stream, sort_keys=True, allow_nan=False)
            temporary.replace(self.path)
        except (OSError, ValueError, TypeError):
            pass

class Miner:
    def __init__(self, identity, manifest, checkpoint, capability=None, state_path=None, progress_path=None):
        check_runtime_profile(manifest)
        for_manifest(manifest)
        self.identity, self.manifest = identity, manifest
        self.cap = capability or identity.decrypt(manifest['capabilities'][identity.id])
        if manifest.get('transport_policy')=='direct-r2-v1':
            from .client import direct_r2_url
            if self.cap.get('transport')!=manifest.get('submission_transport_policy','direct-r2-v1'):raise ValueError('direct R2 upload capability binding')
            direct_r2_url(self.cap.get('put_url'))
            if manifest.get('submission_transport_policy'):
                if len(self.cap.get('batch_put_urls',[]))!=manifest['max_batches']:raise ValueError('bound per-batch upload slots')
                for url in self.cap['batch_put_urls']:direct_r2_url(url)
        entries(manifest)
        self.checkpoint=checkpoint;self.runtime=None;self.runtimes={}
        self.state_path = Path(state_path) if state_path else None
        self.batches = unpack(self.state_path.read_bytes(),budget=for_manifest(manifest)) if self.state_path and self.state_path.exists() else []
        self.progress = MinerProgress(progress_path, manifest) if progress_path else None
        self._progress('epoch_start')
        if any(b['epoch'] != manifest['epoch'] or b['checkpoint'] != manifest['checkpoint']['id'] for b,_ in self.batches):
            raise ValueError('stale local miner state')

    def _progress(self, event, **fields):
        observer = getattr(self, "progress", None)
        if observer is not None: observer.record(event, batches=len(self.batches), **fields)

    def search(self, index, seed=0, max_attempts=100, env_id=None):
        contract=self.manifest.get('sampling_contract')
        if contract is not None:
            if type(seed)is not int or not 0<=seed<contract['max_attempts']:
                raise ValueError('forced sampling search attempt start')
            if type(max_attempts)is not int or max_attempts<1:
                raise ValueError('forced sampling search budget')
            max_attempts=min(max_attempts,contract['max_attempts']-seed)
        if time.time()>=self.manifest.get('deadline',float('inf')):
            self._progress('epoch_deadline')
            raise EpochClosed('signed epoch window closed')
        definition = entry(self.manifest, env_id)
        env_id = definition["env_id"]
        self._progress("task_start", env_id=env_id, index=index)
        resolved=harness_for(definition,index)
        key=(env_id,index,hashlib.sha256(canonical(resolved)).hexdigest())
        if key not in self.runtimes:
            if self.runtime is None:
                self.runtime=make_runtime(self.checkpoint,self.manifest,definition['spec'],resolved)
                self.runtimes[key]=self.runtime
            else:self.runtimes[key]=self.runtime.for_environment(definition['spec'],resolved)
        runtime=self.runtimes[key]
        positive, negative, arrays_pos, arrays_neg = [], [], [], []
        for attempt in range(max_attempts):
            if time.time()>=self.manifest.get('deadline',float('inf')):
                self._progress('task_deadline', env_id=env_id, index=index)
                raise EpochClosed('signed epoch window closed')
            self._progress("attempt_start", env_id=env_id, index=index, attempt=seed+attempt)
            started = time.monotonic()
            try:
                rollout, arrays = runtime.rollout(index, seed+attempt)
            except TaskError as error:
                self._progress("attempt_end", env_id=env_id, index=index, attempt=seed+attempt, outcome="indeterminate", elapsed=time.monotonic()-started, error_type=type(error).__name__)
                # No completed native outcome exists: neither a negative sample nor fraud.
                self.last_generation_error = dict(kind='unscorable_native_task_error',
                    env_id=env_id,index=index,seed=seed+attempt,error_type=type(error).__name__)
                continue
            except Exception as error:
                self._progress("attempt_end", env_id=env_id, index=index, attempt=seed+attempt, outcome="error", elapsed=time.monotonic()-started, error_type=type(error).__name__)
                self._progress("task_error", env_id=env_id, index=index)
                raise
            if time.time()>=self.manifest.get('deadline',float('inf')):
                self._progress('attempt_end', env_id=env_id, index=index, attempt=seed+attempt, outcome='deadline', elapsed=time.monotonic()-started)
                self._progress('task_deadline', env_id=env_id, index=index)
                raise EpochClosed('rollout completed after signed deadline')
            kind = classification(rollout)
            self._progress("attempt_end", env_id=env_id, index=index, attempt=seed+attempt, outcome=kind, elapsed=time.monotonic()-started)
            if kind == 'neutral':
                continue
            dst, arr = (positive, arrays_pos) if kind == 'positive' else (negative, arrays_neg)
            required = self.manifest['K'] if kind == 'positive' else self.manifest['L']
            if len(dst) < required and all(r['turns'] != rollout['turns'] for r in dst):
                dst.append(rollout); arr.append(arrays)
            if len(positive) == self.manifest['K'] and len(negative) == self.manifest['L']:
                batch = dict(schema=2,epoch=self.manifest['epoch'], checkpoint=self.manifest['checkpoint']['id'],
                             env_id=env_id, environment_version=runtime.spec.version, sample_index=index, index=index, rollouts=positive+negative)
                candidate = self.batches + [(batch, arrays_pos+arrays_neg)]
                # A rejected addition must not poison previously uploaded state.
                pack(candidate,budget=for_manifest(self.manifest))
                if time.time()>=self.manifest.get('deadline',float('inf')):
                    self._progress('task_deadline', env_id=env_id, index=index)
                    raise EpochClosed('batch construction completed after signed deadline')
                self.batches = candidate
                self._progress("batch_complete", env_id=env_id, index=index)
                return batch
        self._progress('task_exhausted', env_id=env_id, index=index)
        raise RuntimeError('search budget exhausted')

    def upload(self):
        if time.time()>=self.manifest.get('deadline',float('inf')):
            raise EpochClosed('signed epoch upload window closed')
        data = pack(self.batches,budget=for_manifest(self.manifest))
        if self.state_path:
            self.state_path.parent.mkdir(parents=True,exist_ok=True)
            temporary = self.state_path.with_suffix('.tmp');temporary.write_bytes(data);temporary.chmod(0o600);temporary.replace(self.state_path)
        if time.time()>=self.manifest.get('deadline',float('inf')):
            raise EpochClosed('upload preparation completed after signed deadline')
        if self.manifest.get('submission_transport_policy'):
            from .commitment_transport import VERSION,make,canonical,pair_artifact,UploadJournal
            if self.manifest['submission_transport_policy']!=VERSION:raise ValueError('unsupported commitment upload')
            journal=getattr(self,'_commitment_upload_journal',None)
            if journal is None:
                journal=UploadJournal(self.manifest,self.state_path.with_suffix('.commitment-upload.json')if self.state_path else None);self._commitment_upload_journal=journal
            packed=[(batch,pair_artifact(batch,arrays,self.manifest))for batch,arrays in self.batches]
            for slot,(_,body)in enumerate(packed):
                if journal.known(slot,body):continue
                if time.time()>=self.manifest['deadline']:raise EpochClosed('batch upload deadline')
                response=requests.put(self.cap['batch_put_urls'][slot],data=body,headers=self.cap.get('headers',{}),timeout=120);response.raise_for_status();journal.acknowledge(slot,body)
            if time.time()>=self.manifest['deadline']:raise EpochClosed('commitment upload deadline')
            data=canonical(make(self.identity,self.manifest,packed))
        result = requests.put(self.cap['put_url'], data=data,headers=self.cap.get('headers',{}),timeout=120)
        if result.status_code==403 and time.time()>=self.manifest.get('deadline',float('inf')):
            raise EpochClosed('upload capability expired during request')
        result.raise_for_status()
        return result.status_code
