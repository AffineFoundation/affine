"""Opt-in independent B1 MiMo replicas behind the existing Miner interface.

Spawned workers own independent HF wrappers/models/caches; the parent backend
does no generation so cancellation can terminate every inference process. The source, sampler and token geometry
stay identical. The ordinary Miner still grades group eligibility, journals the
ordered attempts and uploads the original complete group. This adapter is not
enabled implicitly and requires full-rollout qualification before deployment.
"""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import multiprocessing
from pathlib import Path
import shutil
import threading
import time

from .complete_group_parallel import NonceSpool, collect
from .complete_group_codec import encode, decode, limits
from .complete_group_mining import _rollout_from_runtime, digest, task_key

_CHILD = None
_PROOF_GATE = None


def _deadline(manifest):
    if time.time() >= manifest['deadline']:
        from .miner import EpochClosed
        raise EpochClosed('original signed parallel collection window closed')


def _initialize_child(checkpoint, manifest, miner, environment, harness, proof_gate):
    global _CHILD, _PROOF_GATE
    from .complete_group_runtime import MimoGroupRuntime
    _deadline(manifest)
    _CHILD = MimoGroupRuntime(checkpoint, manifest, environment, harness, miner=miner)
    _PROOF_GATE = proof_gate


def _generate(base, manifest, proof_gate, environment, harness, index, nonce):
    _deadline(manifest)
    facade = base.for_environment(environment, harness)
    forward, compute = facade.next_token_logits, facade.compute
    def checked_forward(*args, **kwargs):
        _deadline(manifest)
        return forward(*args, **kwargs)
    def bounded_compute(*args, **kwargs):
        # Full-vocabulary proof prefill has a larger temporary allocation than
        # decoding; serialize this phase while independent B1 decode continues.
        while not proof_gate.acquire(timeout=.5): _deadline(manifest)
        try:
            _deadline(manifest)
            return compute(*args, **kwargs)
        finally: proof_gate.release()
    facade.next_token_logits, facade.compute = checked_forward, bounded_compute
    row, arrays = facade.rollout(index, nonce)
    _deadline(manifest)
    row = _rollout_from_runtime(facade, row, nonce)
    return row, arrays, deepcopy(facade.complete_group_sample_evidence)


def _child_generate(environment, harness, index, nonce):
    return _generate(_CHILD, _CHILD.manifest, _PROOF_GATE, environment, harness, index, nonce)


class _Replicas:
    def __init__(self, base, actor, replicas, state_root):
        self.base, self.actor, self.n = base, actor, replicas
        self.root = Path(state_root).absolute()
        if self.root.resolve() != self.root:
            raise ValueError('ordinary isolated parallel miner state')
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = threading.Lock(); self.closed = False; self.tasks = set()
        self.bound = limits(actor.manifest)
        self.mp = multiprocessing.get_context('spawn')
        self.gate = self.mp.Semaphore(1)
        self.processes = ProcessPoolExecutor(max_workers=replicas, mp_context=self.mp,
            initializer=_initialize_child, initargs=(str(actor.checkpoint), actor.manifest,
                actor.identity.id, base.env_config, base.harness, self.gate))
    def _retire(self):
        for key in tuple(self.tasks):
            if self.actor._group_journal.status(key) in ('accepted','filtered','native_error'):
                path = self.root/key
                if path.exists():
                    if path.is_symlink() or path.resolve() != path:
                        raise ValueError('owned terminal attempt spool')
                    shutil.rmtree(path)
                self.tasks.remove(key)
    def close(self):
        if not self.closed:
            self.closed = True
            try:
                stop_process_pool(self.processes)
            finally: self._retire()
    def _encode(self, nonce, result):
        row, arrays, witness = result
        if row.get('seed') != nonce or witness.get('attempt') != nonce:
            raise ValueError('replica returned a different prescribed nonce')
        return encode([(dict(rollout=row, witness=witness), [arrays])], self.bound)
    def _decode(self, raw):
        rows=decode(raw,self.bound)
        if len(rows)!=1 or set(rows[0][0])!={'rollout','witness'} or len(rows[0][1])!=1:
            raise ValueError('original bounded replica result')
        value,arrays=rows[0]
        return value['rollout'],arrays[0],value['witness']
    def group(self, facade, index):
        if self.closed or not self.lock.acquire(blocking=False):
            raise ValueError('one original group coordinator per replica set')
        spool = None
        try:
            _deadline(self.actor.manifest); self._retire()
            key=task_key(self.actor,facade.spec.id,index);self.tasks.add(key)
            binding=dict(task_key=key,manifest_sha256=digest(self.actor.manifest),
                         miner=self.actor.identity.id,epoch=self.actor.manifest['epoch'])
            spool=NonceSpool(self.root/key,binding,self.actor.batch_contract.group_size,
                             max_attempt_bytes=min(self.bound.compressed_bytes,128*1024**2))
            # Reuse an existing ordinary Miner prefix after interruption instead
            # of generating those completed attempts again.
            for nonce,(row,arrays) in enumerate(self.actor._group_journal.begin(key)):
                witness=row['turns'][0]['behavior_sampling']
                encoded=self._encode(nonce,(row,arrays,witness))
                if spool.has(nonce):
                    import numpy as np
                    old,old_arrays,old_witness=self._decode(spool.read(nonce))
                    if (old!=row or old_witness!=witness or len(old_arrays)!=len(arrays) or
                            any(not np.array_equal(a,b)for a,b in zip(old_arrays,arrays))):
                        raise ValueError('normal journal and parallel original attempt disagree')
                else:spool.put(nonce,encoded)
            def submit(nonce):
                return self.processes.submit(_child_generate,facade.env_config,facade.harness,index,nonce)
            raw=collect(spool,submit,self._encode,workers=self.n,deadline=self.actor.manifest['deadline'],
                        abort=self.close,cancelled=lambda:self.closed)
            return [self._decode(value)for value in raw]
        finally:
            if spool is not None:spool.close()
            self.lock.release()


class ParallelMimoGroupRuntime:
    def __init__(self, base, replicas):
        self._base,self._replicas=base,replicas
        self._completed_index=None;self._completed=None
        self.complete_group_sample_evidence=None
    def __getattr__(self,name):return getattr(self._base,name)
    def for_environment(self,environment,harness=None):
        return type(self)(self._base.for_environment(environment,harness),self._replicas)
    def rollout(self,index,nonce):
        if type(nonce)is not int or not 0<=nonce<self._replicas.actor.batch_contract.group_size:
            raise ValueError('original canonical nonce required')
        if self._completed_index!=index:
            self._completed=self._replicas.group(self._base,index)
            self._completed_index=index
        row,arrays,witness=self._completed[nonce]
        self.complete_group_sample_evidence=deepcopy(witness)
        return deepcopy(row),[a.copy()for a in arrays]
    def close(self):self._replicas.close()


def wrap_runtime(runtime, *, actor, replicas=1, state_root=None, admit=True):
    """CLI/worker local opt-in; default1 returns the unchanged runtime.

    Call once when the actual MiMo actor creates its first MimoGroupRuntime.
    Reuse for_environment views thereafter and close the adapter with the Miner.
    """
    if type(replicas)is not int or replicas not in (1,2,4):
        raise ValueError('explicit one, two or four independent B1 replicas')
    if replicas==1:return runtime
    from .complete_group_runtime import MimoGroupRuntime
    if not isinstance(runtime,MimoGroupRuntime) or runtime.backend.trainable:
        raise ValueError('actual read-only MiMo runtime required')
    if runtime.harness['max_output_tokens']>8192:
        raise ValueError('parallel full-prefill memory profile supports only up to8192')
    runtime.backend.guard()
    if admit:
        import torch
        free,_=torch.cuda.mem_get_info(runtime.device)
        model_bytes=sum(p.numel()*p.element_size()for p in runtime.model.parameters())
        required=replicas*model_bytes+32*1024**3+replicas*2*1024**3
        if free<required:raise RuntimeError('insufficient GPU room for bounded B1 replicas plus serialized proof prefill')
    if state_root is None:
        if actor.state_path is None:raise ValueError('durable ordinary Miner state required for parallel attempts')
        state_root=Path(str(actor.state_path)+'.parallel')
    return ParallelMimoGroupRuntime(runtime,_Replicas(runtime,actor,replicas,state_root))


def stop_process_pool(pool, *, grace_seconds=2.):
    """Bounded owner cancellation, including an in-flight CUDA call.

    Python 3.12 has no public terminate_workers API. Snapshot the executor's own
    children before shutdown, then terminate/kill only those process objects.
    Completed bytes were persisted by the sole coordinator before this call.
    """
    processes=tuple((getattr(pool,'_processes',None) or {}).values())
    pool.shutdown(wait=False,cancel_futures=True)
    for process in processes:
        if process.is_alive(): process.terminate()
    deadline=time.monotonic()+grace_seconds
    for process in processes: process.join(max(0.,deadline-time.monotonic()))
    for process in processes:
        if process.is_alive(): process.kill()
    deadline=time.monotonic()+grace_seconds
    for process in processes: process.join(max(0.,deadline-time.monotonic()))
    if any(process.is_alive()for process in processes):
        raise RuntimeError('replica termination failed; refuse to reuse its worker slot')
