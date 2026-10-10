"""Miner role: decrypt capability, pin checkpoint, search and upload cumulative batches."""
from .forced_sampling import MINER_VERSION
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
    def __init__(self, identity, manifest, checkpoint, capability=None, state_path=None, progress_path=None, search_state_path=None, retire_previous_search=False):
        from .batches import compression_for_manifest
        compression_for_manifest(manifest)
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
                if manifest['submission_transport_policy']in ('small-commitment-pairs-v2','small-commitment-token-pairs-v3'):
                    if len(self.cap.get('training_put_urls',[]))!=manifest['max_batches']:raise ValueError('bound token upload slots')
                    for url in self.cap['training_put_urls']:direct_r2_url(url)
        entries(manifest)
        self.checkpoint=checkpoint;self.runtime=None;self.runtimes={}
        self.state_path = Path(state_path) if state_path else None
        self._prepared_pairs=[];state_bytes=None
        if self.state_path and self.state_path.exists():
            if self.state_path.is_symlink():raise ValueError('private local miner state path')
            with self.state_path.open('rb')as stream:header=stream.read(2)
            if manifest.get('submission_transport_policy') and header!=b'PK':
                from .commitment_transport import read_prepared_state
                self._prepared_pairs=read_prepared_state(self.state_path,manifest)
                self.batches=[(batch,None)for batch,data in self._prepared_pairs]
            else:self.batches=unpack(self.state_path.read_bytes(),budget=for_manifest(manifest))
        else:self.batches=[]
        self.progress = MinerProgress(progress_path, manifest) if progress_path else None
        self._progress('epoch_start')
        if any(b['epoch'] != manifest['epoch'] or b['checkpoint'] != manifest['checkpoint']['id'] for b,_ in self.batches):
            raise ValueError('stale local miner state')
        if manifest.get('sampling_contract',{}).get('version')==MINER_VERSION:
            from .sampling_uniqueness import validate_batch
            if len(self.batches)>manifest['max_batches']:raise ValueError('restored batches exceed signed per-UID limit')
            for batch,_ in self.batches:validate_batch(batch,manifest,identity.id)
        self.search_state = None
        if manifest.get('sampling_contract', {}).get('version') == MINER_VERSION:
            from .miner_search_state import SearchState
            search_path = search_state_path or (self.state_path.with_suffix('.search.sqlite3') if self.state_path else None)
            self.search_state = SearchState(search_path, manifest, identity.id, retire_previous=retire_previous_search)
            try:
                self._retire_completed_searches()
            except BaseException:
                self.close()
                raise

    def _prepared(self):
        from .commitment_transport import pair_artifact,check_prepared_cumulative
        prepared=list(getattr(self,'_prepared_pairs',[]))
        if len(prepared)>len(self.batches):raise ValueError('prepared local batch count')
        for slot,(batch,arrays)in enumerate(self.batches):
            if slot<len(prepared):
                if prepared[slot][0]!=batch:raise ValueError('prepared local batch replacement')
            else:prepared.append((batch,pair_artifact(batch,arrays,self.manifest)))
        check_prepared_cumulative(prepared,self.manifest,len(self.cap['batch_put_urls']))
        self._prepared_pairs=prepared
        return prepared

    def _progress(self, event, **fields):
        observer = getattr(self, "progress", None)
        if observer is not None: observer.record(event, batches=len(self.batches), **fields)

    def close(self):
        journal = getattr(self, 'search_state', None)
        if journal is not None:
            journal.close()
            self.search_state = None
        self._closed = True

    def search(self, index, seed=0, max_attempts=100, env_id=None):
        if getattr(self, '_closed', False):
            raise ValueError('miner is closed')
        journal = getattr(self, 'search_state', None)
        if journal is None:
            return self._legacy_search(index, seed=seed, max_attempts=max_attempts, env_id=env_id)
        with journal.locked():
            return self._resumable_search(index, seed, max_attempts, env_id)

    def _resumable_search(self, index, seed, max_attempts, env_id):
        from .miner_search_state import NoncesExhausted
        from .sampling_uniqueness import validate_batch
        journal = self.search_state
        if type(seed) is not int or not 0 <= seed < journal.maximum:
            raise ValueError('forced sampling search attempt start')
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError('forced sampling search budget')
        if time.time() >= self.manifest['deadline']:
            raise EpochClosed('signed epoch window closed')
        definition = entry(self.manifest, env_id)
        env_id = definition['env_id']
        for batch, _ in self.batches:
            if batch['env_id'] == env_id and batch['index'] == index:
                return batch
        _, completed, rolls, arrays = journal.load(env_id, index)
        if completed:
            raise ValueError('completed search journal missing durable batch state')
        self._progress('task_start', env_id=env_id, index=index)
        runtime = None
        # A fully collected local group survives a crash before upload. Rebuild
        # its complete artifact without consuming another nonce or GPU forward.
        for work in range(max_attempts + 1):
            positives = [(r, a) for r, a in zip(rolls, arrays) if r['classification'] == 'positive']
            negatives = [(r, a) for r, a in zip(rolls, arrays) if r['classification'] == 'negative']
            if len(positives) == self.manifest['K'] and len(negatives) == self.manifest['L']:
                ordered = positives + negatives
                batch = dict(schema=2, epoch=self.manifest['epoch'], checkpoint=self.manifest['checkpoint']['id'],
                             env_id=env_id, environment_version=definition['spec']['version'],
                             sample_index=index, index=index, rollouts=[r for r, _ in ordered])
                validate_batch(batch, self.manifest, self.identity.id)
                candidate = self.batches + [(batch, [a for _, a in ordered])]
                if self.manifest.get('submission_transport_policy'):
                    from .commitment_transport import pair_artifact, check_prepared_cumulative
                    prepared = self._prepared() + [(batch, pair_artifact(batch, candidate[-1][1], self.manifest))]
                    check_prepared_cumulative(prepared, self.manifest, len(self.cap['batch_put_urls']))
                else:
                    pack(candidate, budget=for_manifest(self.manifest))
                if time.time() >= self.manifest['deadline']:
                    raise EpochClosed('batch construction completed after signed deadline')
                if self.manifest.get('submission_transport_policy'):
                    self._prepared_pairs = prepared
                    self.batches = [(b, None) for b, a in candidate]
                else:
                    self.batches = candidate
                self._progress('batch_complete', env_id=env_id, index=index)
                return batch
            if work == max_attempts:
                break
            if time.time() >= self.manifest['deadline']:
                raise EpochClosed('signed epoch window closed')
            # Reserve durably before generation; retries cannot silently repeat
            # an interrupted draw. Explicit seed is a lower bound, never rewind.
            attempt = journal.reserve(env_id, index, seed)
            if runtime is None:
                resolved = harness_for(definition, index)
                key = (env_id, index, hashlib.sha256(canonical(resolved)).hexdigest())
                if key not in self.runtimes:
                    if self.runtime is None:
                        self.runtime = make_runtime(self.checkpoint, self.manifest, definition['spec'], resolved, miner=self.identity.id)
                        self.runtimes[key] = self.runtime
                    else:
                        self.runtimes[key] = self.runtime.for_environment(definition['spec'], resolved)
                runtime = self.runtimes[key]
            self._progress('attempt_start', env_id=env_id, index=index, attempt=attempt)
            started = time.monotonic()
            try:
                rollout, proof_arrays = runtime.rollout(index, attempt)
            except TaskError as error:
                self._progress('attempt_end', env_id=env_id, index=index, attempt=attempt, outcome='indeterminate', elapsed=time.monotonic()-started, error_type=type(error).__name__)
                self.last_generation_error = dict(kind='unscorable_native_task_error', env_id=env_id, index=index, seed=attempt, error_type=type(error).__name__)
                continue
            except Exception as error:
                self._progress('attempt_end', env_id=env_id, index=index, attempt=attempt, outcome='error', elapsed=time.monotonic()-started, error_type=type(error).__name__)
                raise
            if time.time() >= self.manifest['deadline']:
                raise EpochClosed('rollout completed after signed deadline')
            kind = classification(rollout)
            self._progress('attempt_end', env_id=env_id, index=index, attempt=attempt, outcome=kind, elapsed=time.monotonic()-started)
            if kind != 'neutral':
                rolls, arrays = journal.accept(env_id, index, rollout, proof_arrays, attempt)
        self._progress('task_exhausted', env_id=env_id, index=index)
        raise RuntimeError('search budget exhausted; partial progress retained')

    def _retire_completed_searches(self):
        journal = getattr(self, 'search_state', None)
        if journal is not None and self.state_path:
            with journal.locked():
                journal.completed(self.batches)

    def _legacy_search(self, index, seed=0, max_attempts=100, env_id=None):
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
                self.runtime=make_runtime(self.checkpoint,self.manifest,definition['spec'],resolved,**({'miner':self.identity.id}if contract and contract['version']==MINER_VERSION else {}))
                self.runtimes[key]=self.runtime
            else:self.runtimes[key]=self.runtime.for_environment(definition['spec'],resolved)
        runtime=self.runtimes[key]
        positive, negative, arrays_pos, arrays_neg = [], [], [], []
        from .sampling_uniqueness import validate_batch, content_digest
        fingerprints=set()
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
            v5=contract and contract['version']==MINER_VERSION
            signature=content_digest(rollout)if v5 else None
            distinct=signature not in fingerprints if v5 else all(r['turns']!=rollout['turns']for r in dst)
            if len(dst) < required and distinct:
                dst.append(rollout); arr.append(arrays);fingerprints.add(signature)
            if len(positive) == self.manifest['K'] and len(negative) == self.manifest['L']:
                batch = dict(schema=2,epoch=self.manifest['epoch'], checkpoint=self.manifest['checkpoint']['id'],
                             env_id=env_id, environment_version=runtime.spec.version, sample_index=index, index=index, rollouts=positive+negative)
                if v5:validate_batch(batch,self.manifest,self.identity.id)
                candidate = self.batches + [(batch, arrays_pos+arrays_neg)]
                # A rejected addition must not poison previously uploaded state.
                if self.manifest.get('submission_transport_policy'):
                    from .commitment_transport import pair_artifact,check_prepared_cumulative
                    prepared=self._prepared()+[(batch,pair_artifact(batch,arrays_pos+arrays_neg,self.manifest))]
                    check_prepared_cumulative(prepared,self.manifest,len(self.cap['batch_put_urls']))
                else:pack(candidate,budget=for_manifest(self.manifest))
                if time.time()>=self.manifest.get('deadline',float('inf')):
                    self._progress('task_deadline', env_id=env_id, index=index)
                    raise EpochClosed('batch construction completed after signed deadline')
                if self.manifest.get('submission_transport_policy'):
                    self._prepared_pairs=prepared;self.batches=[(b,None)for b,a in candidate]
                else:self.batches=candidate
                self._progress("batch_complete", env_id=env_id, index=index)
                return batch
        self._progress('task_exhausted', env_id=env_id, index=index)
        raise RuntimeError('search budget exhausted')

    def upload(self):
        if time.time()>=self.manifest.get('deadline',float('inf')):
            raise EpochClosed('signed epoch upload window closed')
        if self.manifest.get('submission_transport_policy'):
            from .commitment_transport import VERSION2,VERSION3,VERSIONS,make,canonical,UploadJournal,write_prepared_state
            if self.manifest['submission_transport_policy']not in VERSIONS:raise ValueError('unsupported commitment upload')
            packed=self._prepared()
            if self.state_path:
                write_prepared_state(self.state_path,self.manifest,packed)
                self._retire_completed_searches()
            journal=getattr(self,'_commitment_upload_journal',None)
            if journal is None:
                journal=UploadJournal(self.manifest,self.state_path.with_suffix('.commitment-upload.json')if self.state_path else None);self._commitment_upload_journal=journal
            for slot,(_,body)in enumerate(packed):
                if journal.known(slot,body):continue
                remaining=min(120,self.manifest['deadline']-time.time()-1)
                if remaining<=0:raise EpochClosed('batch upload deadline')
                response=requests.put(self.cap['batch_put_urls'][slot],data=body,headers=self.cap.get('headers',{}),timeout=remaining);response.raise_for_status();journal.acknowledge(slot,body)
            if self.manifest['submission_transport_policy']in (VERSION2,VERSION3):
                from .training_documents import document
                for slot,(batch,_)in enumerate(packed):
                    body=document(batch,self.manifest,self.identity.id,slot)
                    if journal.known('training-'+str(slot),body):continue
                    remaining=min(120,self.manifest['deadline']-time.time()-1)
                    if remaining<=0:raise EpochClosed('token document upload deadline')
                    response=requests.put(self.cap['training_put_urls'][slot],data=body,headers=self.cap.get('headers',{}),timeout=remaining,allow_redirects=False);response.raise_for_status();journal.acknowledge('training-'+str(slot),body)
            if time.time()>=self.manifest['deadline']:raise EpochClosed('commitment upload deadline')
            data=canonical(make(self.identity,self.manifest,packed))
        else:
            data=pack(self.batches,budget=for_manifest(self.manifest))
            if self.state_path:
                self.state_path.parent.mkdir(parents=True,exist_ok=True)
                temporary=self.state_path.with_suffix('.tmp');temporary.write_bytes(data);temporary.chmod(0o600);temporary.replace(self.state_path)
                self._retire_completed_searches()
        now=time.time()
        if now>=self.manifest.get('deadline',float('inf')):raise EpochClosed('upload preparation completed after signed deadline')
        timeout=min(120,max(.001,self.manifest['deadline']-now-1))if self.manifest.get('submission_transport_policy')else 120
        result = requests.put(self.cap['put_url'], data=data,headers=self.cap.get('headers',{}),timeout=timeout)
        if result.status_code==403 and time.time()>=self.manifest.get('deadline',float('inf')):
            raise EpochClosed('upload capability expired during request')
        result.raise_for_status()
        return result.status_code
