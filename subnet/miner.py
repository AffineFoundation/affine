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

class Miner:
    def __init__(self, identity, manifest, checkpoint, capability=None, state_path=None):
        check_runtime_profile(manifest)
        for_manifest(manifest)
        self.identity, self.manifest = identity, manifest
        self.cap = capability or identity.decrypt(manifest['capabilities'][identity.id])
        if manifest.get('transport_policy')=='direct-r2-v1':
            from .client import direct_r2_url
            if self.cap.get('transport')!='direct-r2-v1':raise ValueError('direct R2 upload capability binding')
            direct_r2_url(self.cap.get('put_url'))
        entries(manifest)
        self.checkpoint=checkpoint;self.runtime=None;self.runtimes={}
        self.state_path = Path(state_path) if state_path else None
        self.batches = unpack(self.state_path.read_bytes(),budget=for_manifest(manifest)) if self.state_path and self.state_path.exists() else []
        if any(b['epoch'] != manifest['epoch'] or b['checkpoint'] != manifest['checkpoint']['id'] for b,_ in self.batches):
            raise ValueError('stale local miner state')

    def search(self, index, seed=0, max_attempts=100, env_id=None):
        definition = entry(self.manifest, env_id)
        env_id = definition["env_id"]
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
            if time.time()>=self.manifest.get('deadline',float('inf')):break
            try:
                rollout, arrays = runtime.rollout(index, seed+attempt)
            except TaskError as error:
                # No completed native outcome exists: neither a negative sample nor fraud.
                self.last_generation_error = dict(kind='unscorable_native_task_error',
                    env_id=env_id,index=index,seed=seed+attempt,error_type=type(error).__name__)
                continue
            kind = classification(rollout)
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
                self.batches = candidate
                return batch
        raise RuntimeError('search budget exhausted')

    def upload(self):
        data = pack(self.batches,budget=for_manifest(self.manifest))
        if self.state_path:
            self.state_path.parent.mkdir(parents=True,exist_ok=True)
            temporary = self.state_path.with_suffix('.tmp');temporary.write_bytes(data);temporary.chmod(0o600);temporary.replace(self.state_path)
        result = requests.put(self.cap['put_url'], data=data,headers=self.cap.get('headers',{}),timeout=120)
        result.raise_for_status()
        return result.status_code
