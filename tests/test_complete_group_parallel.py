"""CPU executor/journal controls, not CUDA concurrency qualification."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

import pytest

from subnet.complete_group_parallel import NonceSpool, collect


def encode(nonce, value):
    assert value['nonce'] == nonce
    return json.dumps(value, sort_keys=True).encode()


def test_out_of_order_attempts_are_saved_and_returned_in_nonce_order(tmp_path):
    spool = NonceSpool(tmp_path/'spool', {'checkpoint':'original','task':12}, 8)
    completed = []
    def generate(nonce):
        time.sleep(.01 if nonce % 2 else .04)
        return dict(nonce=nonce, tokens=[nonce, 42], outcome='wrong' if nonce % 2 else 'correct')
    try:
        with ThreadPoolExecutor(2) as pool:
            rows = collect(spool, lambda n:pool.submit(generate,n), encode, workers=2,
                           progress=lambda nonce,count:completed.append(nonce))
        assert [json.loads(v)['nonce'] for v in rows] == list(range(8))
        assert completed.index(1) < completed.index(0)
        with ThreadPoolExecutor(2) as pool:
            assert collect(spool, lambda n:pytest.fail('completed attempt repeated'), encode, workers=2) == rows
    finally: spool.close()


def test_failed_prefix_retains_other_original_nonces_and_retry_fills_only_gap(tmp_path):
    root = tmp_path/'spool'; binding = {'task':1,'checkpoint':'original'}
    spool = NonceSpool(root, binding, 8)
    barrier = threading.Barrier(2)
    def generate(nonce):
        barrier.wait(timeout=2)
        if nonce == 0: raise RuntimeError('infrastructure error')
        time.sleep(.02)
        return dict(nonce=nonce, tokens=[7,8])
    try:
        with ThreadPoolExecutor(2) as pool, pytest.raises(RuntimeError, match='infrastructure'):
            collect(spool, lambda n:pool.submit(generate,n), encode, workers=2)
        assert not spool.has(0) and spool.has(1)
        with pytest.raises(ValueError, match='all N'): spool.ordered()
    finally: spool.close()
    spool = NonceSpool(root, binding, 8); called=[]
    try:
        with ThreadPoolExecutor(2) as pool:
            def fixed(n): called.append(n); return dict(nonce=n,tokens=[7,8])
            rows=collect(spool,lambda n:pool.submit(fixed,n),encode,workers=2)
        assert sorted(called)==[0,2,3,4,5,6,7]
        assert [json.loads(r)['nonce']for r in rows]==list(range(8))
        with pytest.raises(ValueError,match='cannot change'):spool.put(1,b'changed')
    finally:spool.close()
    with pytest.raises(ValueError,match='binding changed'):NonceSpool(root,{'task':2},8)


def test_late_completion_after_deadline_is_saved_but_not_returned_as_group(tmp_path):
    spool=NonceSpool(tmp_path/'spool',{'task':1},8)
    try:
        def generate(n):time.sleep(.04);return dict(nonce=n,tokens=[3])
        with ThreadPoolExecutor(2)as pool,pytest.raises(TimeoutError):
            collect(spool,lambda n:pool.submit(generate,n),encode,workers=2,deadline=time.time()+.01)
        assert set(spool.state['attempts'])=={'0','1'}
        assert not spool.has(2)
    finally:spool.close()


def test_spool_refuses_tampering_and_duplicate_coordinator(tmp_path):
    root=tmp_path/'spool';spool=NonceSpool(root,{'task':1},8)
    try:
        with pytest.raises(BlockingIOError):NonceSpool(root,{'task':1},8)
        spool.put(0,b'original')
    finally:spool.close()
    (root/'0.bin').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='digest changed'):NonceSpool(root,{'task':1},8)


def test_real_attempt_journal_advances_only_contiguous_original_prefix(tmp_path):
    import numpy as np
    from subnet.complete_group_state import AttemptJournal
    from subnet.complete_group_parallel import append_original_prefix
    from test_complete_group_fixtures import fixture
    manifest=fixture();spool=NonceSpool(tmp_path/'spool',{'task':1},8)
    journal=AttemptJournal(tmp_path/'actor.json',manifest,'a'*64);key='b'*64
    def payload(n):return json.dumps(dict(row=dict(seed=n),arrays=[[[-float(n)]]])).encode()
    def decode(raw):
        value=json.loads(raw)
        return value['row'],[np.asarray(a,dtype=np.float32)for a in value['arrays']]
    try:
        spool.put(1,payload(1))
        assert append_original_prefix(spool,journal,key,decode)==[]
        spool.put(0,payload(0))
        rows=append_original_prefix(spool,journal,key,decode)
        assert [r['seed']for r,_ in rows]==[0,1]
        assert len(append_original_prefix(spool,journal,key,decode))==2
        for n in range(2,8):spool.put(n,payload(n))
        assert [r['seed']for r,_ in append_original_prefix(spool,journal,key,decode)]==list(range(8))
    finally:journal.close();spool.close()
