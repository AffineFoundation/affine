"""Actual Miner/search/journal/codec integration with explicit CPU workers.

The executor substitution here tests orchestration, never asserts CUDA model
parity. Separate H200 prefix/full-rollout records qualify the actual replicas.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from types import SimpleNamespace
from unittest.mock import patch
import time

import pytest

from subnet.miner import Miner
from subnet.storage import Identity
from subnet.complete_group_transport import CompleteGroupTransport, capability
from subnet.mimo_parallel_mining import ParallelMimoGroupRuntime, _Replicas
from test_complete_group_mimo import mimo_manifest, MimoCollectionRuntime
from test_complete_group_transport import FakeBucket, FakeHTTP


def test_replica_admission_uses_actual_facade_backend(tmp_path,monkeypatch):
    import torch
    import subnet.mimo_parallel_mining as parallel
    from subnet.complete_group_runtime import MimoGroupRuntime
    facade=object.__new__(MimoGroupRuntime)
    guarded=[];devices=[]
    facade.backend=SimpleNamespace(trainable=False,device='cuda:1',
        model=torch.nn.Linear(3,2),guard=lambda:guarded.append(True))
    facade.harness={'max_output_tokens':8192}
    actor=SimpleNamespace(state_path=tmp_path/'miner.json')
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(devices.append(device)or(140*1024**3,140*1024**3)))
    monkeypatch.setattr(parallel,'_Replicas',lambda *args:SimpleNamespace(close=lambda:None))
    wrapped=parallel.wrap_runtime(facade,actor=actor,replicas=4)
    assert wrapped._base is facade and guarded==[True] and devices==['cuda:1']
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(1,140*1024**3))
    with pytest.raises(RuntimeError,match='insufficient GPU room'):
        parallel.wrap_runtime(facade,actor=actor,replicas=4)


def test_proof_temporary_buffers_released_inside_shared_memory_gate(monkeypatch):
    import subnet.mimo_parallel_mining as parallel
    events=[]
    class Gate:
        held=False
        def acquire(self,timeout):self.held=True;events.append('acquire');return True
        def release(self):events.append('release');self.held=False
    gate=Gate()
    class Backend:
        complete_group_sample_evidence={'attempt':0}
        def for_environment(self,*args):return self
        def next_token_logits(self,*args):return 'unchanged-logits'
        def compute(self,*args):events.append('compute');return 'original-proof-bytes'
        def rollout(self,index,nonce):return {'proof':self.compute()},[]
    def release():
        assert gate.held
        events.append('empty-inactive-cache')
    monkeypatch.setattr(parallel,'_release_cuda_temporaries',release)
    monkeypatch.setattr(parallel,'_rollout_from_runtime',lambda facade,row,nonce:row)
    row,arrays,witness=parallel._generate(Backend(),{'deadline':time.time()+10},gate,{}, {},0,0)
    assert row=={'proof':'original-proof-bytes'} and arrays==[] and witness=={'attempt':0}
    assert events==['acquire','empty-inactive-cache','compute','empty-inactive-cache','release']


class Facade(MimoCollectionRuntime):
    def __init__(self,manifest,*args,**kwargs):
        super().__init__(manifest,*args,**kwargs)
        self.manifest=manifest
    def for_environment(self,environment,harness):
        result=copy(self);result.complete_group_sample_evidence=None
        return result
    def next_token_logits(self,*args,**kwargs):raise AssertionError('fixture has no GPU forward')
    def compute(self,*args,**kwargs):raise AssertionError('fixture has no GPU prefill')
    def rollout(self,index,nonce):
        time.sleep(.02 if nonce==0 else .001)
        return super().rollout(index,nonce)


def setup_actor(tmp_path,monkeypatch,outcomes=None):
    import subnet.mimo_parallel_mining as parallel
    m=mimo_manifest();m['deadline']=int(time.time())+120
    identity=Identity(bytes([96])*32);bucket=FakeBucket();http=FakeHTTP(bucket)
    m['capabilities']={identity.id:capability(bucket,m,identity.id,registered_miners={identity.id})}
    with patch('subnet.model.check_runtime_profile'):
        actor=Miner(identity,m,'fixture-only',state_path=tmp_path/'actor.json',
                    complete_group_transport=CompleteGroupTransport(http=http))
    base=Facade(m,identity.id,outcomes=outcomes)
    def fake_process_pool(*,max_workers,initializer,initargs,**kwargs):
        checkpoint,manifest,miner,environment,harness,gate=initargs
        parallel._CHILD=Facade(manifest,miner,outcomes=outcomes);parallel._PROOF_GATE=gate
        return ThreadPoolExecutor(max_workers=max_workers)
    monkeypatch.setattr(parallel,'ProcessPoolExecutor',fake_process_pool)
    service=_Replicas(base,actor,2,tmp_path/'replicas')
    wrapper=ParallelMimoGroupRuntime(base,service)
    monkeypatch.setattr('subnet.miner.make_runtime',lambda *a,**kw:wrapper)
    return actor,wrapper,service


def test_real_miner_consumes_all_original_parallel_attempts_then_uploads(tmp_path,monkeypatch):
    actor,wrapper,service=setup_actor(tmp_path,monkeypatch)
    try:
        batch=actor.search(0,max_attempts=8)
        assert [r['seed']for r in batch['rollouts']]==list(range(8))
        assert actor.upload()==200
        assert actor.search(0)==batch and len(actor.batches)==1
        assert all(r['turns'][0]['behavior_sampling']['attempt']==r['seed']for r in batch['rollouts'])
        assert list((tmp_path/'replicas').glob('*/STATE.json'))
    finally:
        wrapper.close();actor._group_journal.close()
    assert not list((tmp_path/'replicas').glob('*/STATE.json'))


def test_complete_all_correct_group_remains_filtered_without_replacement(tmp_path,monkeypatch):
    from subnet.complete_group_mining import GroupFiltered
    actor,wrapper,service=setup_actor(tmp_path,monkeypatch,['positive']*8)
    try:
        with pytest.raises(GroupFiltered):actor.search(0,max_attempts=8)
        assert len(wrapper._completed)==8 and not actor.batches
        with pytest.raises(GroupFiltered):actor.search(0,max_attempts=8)
        assert not actor.batches
    finally:wrapper.close();actor._group_journal.close()


def test_native_error_retires_pool_but_next_distinct_task_can_run(tmp_path,monkeypatch):
    import subnet.mimo_parallel_mining as parallel
    from subnet.complete_group_mining import GroupIndeterminate, task_key
    from verifiers.v1.errors import TaskError
    actor,wrapper,service=setup_actor(tmp_path,monkeypatch)
    old_gate=service.gate
    original=parallel._child_generate
    calls=[]
    def fail_first(environment,harness,index,nonce):
        calls.append((index,nonce))
        if index==0 and nonce==0:
            old_gate.acquire()  # Simulate termination while owning proof gate.
            raise TaskError('native grading unavailable')
        return original(environment,harness,index,nonce)
    monkeypatch.setattr(parallel,'_child_generate',fail_first)
    try:
        with pytest.raises(GroupIndeterminate):actor.search(0,max_attempts=8)
        assert not service.closed and service.processes is None
        assert actor._group_journal.status(task_key(actor,wrapper.spec.id,0))=='native_error'
        before=list(calls)
        with pytest.raises(GroupIndeterminate):actor.search(0,max_attempts=8)
        assert calls==before  # The failed original group cannot be replaced.
        batch=actor.search(1,max_attempts=8)
        assert batch['index']==1 and [r['seed']for r in batch['rollouts']]==list(range(8))
        assert len(actor.batches)==1 and service.processes is not None
        assert service.gate is not old_gate and service.gate.acquire(timeout=.1)
        service.gate.release()
        assert sorted(nonce for index,nonce in calls if index==1)==list(range(8))
    finally:wrapper.close();actor._group_journal.close()
    assert service.actor is None and service.base is None and wrapper._base is None


def blocked_original_attempt(started):
    started.set()
    time.sleep(30)
    return 'never-completed-original'


def test_deadline_terminates_real_process_instead_of_waiting_for_rollout(tmp_path):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from subnet.mimo_parallel_mining import stop_process_pool
    from subnet.complete_group_parallel import NonceSpool, collect
    mp=multiprocessing.get_context('spawn')
    with mp.Manager() as manager:
        began=manager.Event()
        pool=ProcessPoolExecutor(1,mp_context=mp)
        future=pool.submit(blocked_original_attempt,began)
        assert began.wait(15)
        processes=tuple(pool._processes.values())
        spool=NonceSpool(tmp_path/'deadline',{'checkpoint':'original'},8)
        start=time.monotonic()
        try:
            with pytest.raises(TimeoutError):
                collect(spool,lambda nonce:future,lambda nonce,value:value.encode(),workers=1,
                    deadline=time.time()+.1,abort=lambda:stop_process_pool(pool,grace_seconds=.5))
            assert time.monotonic()-start<3
            assert not spool.state['attempts']
            assert all(not p.is_alive()for p in processes)
        finally:spool.close();stop_process_pool(pool,grace_seconds=.5)


def test_close_cancels_real_inference_process_and_is_bounded():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from subnet.mimo_parallel_mining import stop_process_pool
    mp=multiprocessing.get_context('spawn')
    with mp.Manager() as manager:
        began=manager.Event();pool=ProcessPoolExecutor(1,mp_context=mp)
        pool.submit(blocked_original_attempt,began)
        assert began.wait(15)
        processes=tuple(pool._processes.values());start=time.monotonic()
        stop_process_pool(pool,grace_seconds=.5)
        assert time.monotonic()-start<2
        assert all(not p.is_alive()for p in processes)


def test_completed_pool_shuts_down_cleanly_without_signalling_children():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from subnet.mimo_parallel_mining import stop_process_pool
    pool=ProcessPoolExecutor(2,mp_context=multiprocessing.get_context('spawn'))
    try:
        assert [f.result(timeout=15)for f in(pool.submit(abs,-7),pool.submit(abs,-8))]==[7,8]
        processes=tuple(pool._processes.values())
        manager=pool._executor_manager_thread
        stop_process_pool(pool,grace_seconds=1)
        assert all(process.exitcode==0 for process in processes)
        assert not manager.is_alive()
        stop_process_pool(pool,grace_seconds=1)  # Already closed is harmless.
    finally:
        stop_process_pool(pool,grace_seconds=1)


def test_ready_exit_sentinel_wins_over_racing_process_liveness():
    """The executor manager may have reaped an idle child before poll updates."""
    import os
    from subnet.mimo_parallel_mining import stop_process_pool
    reader,writer=os.pipe()
    os.close(writer)  # Real OS exit readiness, independently of Process state.
    calls=[]
    class ReapingChild:
        sentinel=reader
        def is_alive(self):raise AssertionError('do not race the manager waitpid')
        def join(self,*a,**k):raise AssertionError('manager owns child reaping')
        def terminate(self):raise AssertionError('exited child must not be signalled')
        def kill(self):raise AssertionError('exited child must not be signalled')
    manager=SimpleNamespace(join=lambda **kw:calls.append('manager-cleanup'),is_alive=lambda:False)
    pool=SimpleNamespace(_processes={17:ReapingChild()},_executor_manager_thread=manager,
        shutdown=lambda **kw:calls.append(('shutdown',kw)))
    try:
        stop_process_pool(pool,grace_seconds=.01)
        assert calls==[('shutdown',{'wait':False,'cancel_futures':True}),'manager-cleanup']
    finally:os.close(reader)


def test_unconfirmed_exit_does_not_free_replica_slot(monkeypatch):
    import os
    import subnet.mimo_parallel_mining as parallel
    reader,writer=os.pipe()  # An unresponsive child never signals completion.
    signals=[]
    child=SimpleNamespace(sentinel=reader,terminate=lambda:signals.append('term'),
                          kill=lambda:signals.append('kill'))
    pool=SimpleNamespace(_processes={17:child},shutdown=lambda **kw:None)
    replicas=object.__new__(_Replicas);replicas.processes=pool;replicas.closed=False
    original=parallel.stop_process_pool
    monkeypatch.setattr(parallel,'stop_process_pool',lambda pool:original(pool,grace_seconds=.01))
    try:
        with pytest.raises(RuntimeError,match='termination failed'):
            replicas._abort_pool()
        assert signals==['term','kill'] and replicas.closed
        with pytest.raises(ValueError,match='already closed'):replicas._ensure_pool()
    finally:os.close(reader);os.close(writer)
