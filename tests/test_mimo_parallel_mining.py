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
