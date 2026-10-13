"""Actual write/hash receipts cross staging; cold recovery never trusts a memo."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import pytest

from subnet.mimo_grpo import transaction as t
from test_mimo_transaction_hash_memo import fixture, count_hashes, expose_changed_ctime


def shared(callbacks, *, after_hash=None, saved=None):
    old_stage=callbacks['stage']
    def stage(report,root,hasher):
        names=old_stage(report,root)
        for name in names:hasher(root/name)
        # Stage inventory can ask again; only actual full reads seeded receipts.
        for name in names:hasher(root/name)
        if saved is not None:saved.append(hasher)
        if after_hash is not None:after_hash(root,hasher)
        return names
    return dict(update=callbacks['update'],stage_with_hasher=stage,publish_model=callbacks['publish_model'])


def test_stage_receipts_reused_by_both_transaction_inventories(tmp_path,monkeypatch):
    tx,callbacks,_=fixture(tmp_path/'shared');hashes=count_hashes(monkeypatch)
    result=tx.run(**shared(callbacks));assert len(hashes)==3
    other,callbacks,_=fixture(tmp_path/'old');baseline=other.run(**callbacks)
    assert t.canonical(result)==t.canonical(baseline)
    assert len(hashes)==6


@pytest.mark.parametrize('where',['inside_stage','after_stage'])
def test_same_size_changed_bytes_restored_mtime_fail_original_receipt(tmp_path,monkeypatch,where):
    tx,callbacks,_=fixture(tmp_path/'job')
    def mutate(root,hasher):
        path=root/'optimizer/state.bin';before=path.stat()
        path.write_bytes(b'OPTIMIZER');os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
        expose_changed_ctime(monkeypatch,before)
        if where=='inside_stage':hasher(path)
    with pytest.raises(ValueError,match='artifact.*(changed|integrity)'):
        tx.run(**shared(callbacks,after_hash=mutate))
    assert tx._read()['state']=='updating'


def test_replaced_inode_same_bytes_rehashed(tmp_path,monkeypatch):
    tx,callbacks,_=fixture(tmp_path/'job');hashes=count_hashes(monkeypatch)
    def replace(root,hasher):
        path=root/'optimizer/state.bin';replacement=path.with_name('swap')
        replacement.write_bytes(path.read_bytes());replacement.replace(path)
    result=tx.run(**shared(callbacks,after_hash=replace))
    assert result['state']=='published'and len(hashes)==4


@pytest.mark.parametrize('change',['symlink','ancestor_symlink','outside_root','other_thread'])
def test_hasher_rejects_unsafe_path_or_owner(tmp_path,change):
    tx,callbacks,_=fixture(tmp_path/'job')
    def unsafe(root,hasher):
        path=root/'optimizer/state.bin'
        if change=='symlink':
            target=root/'optimizer/target';target.write_bytes(path.read_bytes());path.unlink();path.symlink_to(target)
        elif change=='ancestor_symlink':
            actual=root/'saved';(root/'optimizer').rename(actual);(root/'optimizer').symlink_to(actual)
        elif change=='outside_root':
            path=tmp_path/'outside';path.write_bytes(b'x')
        elif change=='other_thread':
            with ThreadPoolExecutor(max_workers=1)as pool:pool.submit(hasher,path).result()
            return
        hasher(path)
    with pytest.raises(ValueError):tx.run(**shared(callbacks,after_hash=unsafe))


def test_stage_callback_hasher_expires_and_cannot_seed_from_metadata(tmp_path):
    tx,callbacks,_=fixture(tmp_path/'job');saved=[]
    tx.run(**shared(callbacks,saved=saved))
    with pytest.raises(ValueError,match='active staging owner'):saved[0](tx.root/'optimizer/state.bin')
    # The only API accepts a path, not a checksum or serialized cache.
    with pytest.raises(ValueError):saved[0]({'sha256':'a'*64})


@pytest.mark.parametrize('cold',[False,True])
def test_publication_retry_reads_every_byte_and_never_updates_again(tmp_path,monkeypatch,cold):
    tx,callbacks,calls=fixture(tmp_path/'job');hashes=count_hashes(monkeypatch)
    def failed(request):raise TimeoutError('lost acknowledgement')
    with pytest.raises(TimeoutError):tx.run(**dict(shared(callbacks),publish_model=failed))
    assert len(hashes)==3
    if cold:tx=t.Transaction(tx.root,tx.job)
    tx.run(**shared(callbacks))
    assert len(hashes)==6 and calls['update']==calls['stage']==1


def test_cold_retry_detects_tampered_optimizer(tmp_path):
    tx,callbacks,_=fixture(tmp_path/'job')
    def failed(request):raise TimeoutError('lost acknowledgement')
    with pytest.raises(TimeoutError):tx.run(**dict(shared(callbacks),publish_model=failed))
    (tx.root/'optimizer/state.bin').write_bytes(b'OPTIMIZER')
    with pytest.raises(ValueError,match='artifact integrity'):
        t.Transaction(tx.root,tx.job).run(**shared(callbacks))


def test_stage_failure_invalidates_callback_and_requires_explicit_recovery(tmp_path):
    tx,callbacks,calls=fixture(tmp_path/'job');saved=[]
    def failed(root,hasher):saved.append(hasher);raise OSError('interrupted state write')
    with pytest.raises(OSError):tx.run(**shared(callbacks,after_hash=failed))
    with pytest.raises(ValueError,match='active staging owner'):saved[0](tx.root/'optimizer/state.bin')
    with pytest.raises(t.RecoveryRequired):tx.run(**shared(callbacks))
    assert calls['update']==1


def test_actual_cpu_optimizer_state_hashes_once_and_restores(tmp_path,monkeypatch):
    import torch
    from test_mimo_grpo import Runtime,groups,binding,execute
    from subnet.mimo_grpo import TrainerConfig
    from subnet.mimo_grpo.optimizer import CPUAdamW
    torch.set_num_threads(1)
    runtime=Runtime();optimizer=CPUAdamW(runtime.model.named_parameters(),TrainerConfig(),input_checkpoint=runtime.checkpoint_id)
    hashes=count_hashes(monkeypatch)
    result=execute(tmp_path/'job',runtime,optimizer,binding())
    # Actual safetensors save_local shard hashes seed the same guarded receipt
    # used for local_state_files and both transaction inventories, one each.
    assert len(hashes)==len(result['artifacts'])
    descriptor=json.loads((tmp_path/'job/optimizer/descriptor.json').read_bytes())
    restored=CPUAdamW(runtime.model.named_parameters(),TrainerConfig(),input_checkpoint=runtime.checkpoint_id)
    restored.restore_local(tmp_path/'job/optimizer',descriptor_sha256=t.digest(descriptor),checkpoint=runtime.checkpoint_id)
    assert restored.global_step==1 and len(hashes)==len(result['artifacts'])+len(descriptor['chunks'])
    for name in optimizer.masters:
        assert torch.equal(optimizer.masters[name],restored.masters[name])
        for slot in('exp_avg','exp_avg_sq'):
            assert torch.equal(optimizer.optimizer.state[optimizer.masters[name]][slot],restored.optimizer.state[restored.masters[name]][slot])


def test_new_process_rehashes_and_resumes_without_second_update(tmp_path):
    import subprocess,sys
    tx,callbacks,_=fixture(tmp_path/'job')
    def failed(request):raise TimeoutError('lost acknowledgement')
    with pytest.raises(TimeoutError):tx.run(**dict(shared(callbacks),publish_model=failed))
    script='''
import json,sys
from pathlib import Path
from subnet.mimo_grpo import transaction as t
from test_mimo_transaction_hash_memo import fixture
original=t.hashlib.sha256;reads=[]
def measured(*a,**kw):
 if not a and not kw:reads.append(True)
 return original(*a,**kw)
t.hashlib.sha256=measured
tx,callbacks,calls=fixture(Path(sys.argv[1]))
r=tx.run(**callbacks)
print(json.dumps(dict(state=r['state'],hashes=len(reads),calls=calls)))
'''
    result=subprocess.run([sys.executable,'-B','-c',script,str(tx.root)],check=True,capture_output=True,text=True,timeout=30)
    value=json.loads(result.stdout)
    assert value==dict(state='published',hashes=3,calls=dict(update=0,stage=0,publish=1))


def test_caller_owned_receipt_mutation_cannot_seed_or_change_memo(tmp_path,monkeypatch):
    tx,callbacks,_=fixture(tmp_path/'job');hashes=count_hashes(monkeypatch)
    def mutate_receipt(root,hasher):
        path=root/'optimizer/state.bin';row=hasher(path);expected=dict(row)
        row.update(sha256='0'*64,size=999)
        assert hasher(path)==expected
    result=tx.run(**shared(callbacks,after_hash=mutate_receipt))
    assert result['artifacts']['optimizer/state.bin']['size']==9
    assert result['artifacts']['optimizer/state.bin']['sha256']!='0'*64
    assert len(hashes)==3
