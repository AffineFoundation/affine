"""Local parallel opt-in reaches the real Miner; no CUDA qualification here."""
import time
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from test_complete_group_mimo import mimo_manifest, MimoCollectionRuntime
from test_complete_group_transport import FakeBucket, FakeHTTP
from subnet.complete_group_transport import CompleteGroupTransport, capability
from subnet.storage import Identity
from subnet.miner import Miner


@pytest.mark.parametrize('replicas',[True,False,0,3,8,2.,'2'])
def test_invalid_local_replicas_reject_before_contract_or_model(replicas):
    with patch('subnet.miner.make_runtime') as model:
        with pytest.raises(ValueError,match='replicas'):
            Miner(None,{},None,mimo_replicas=replicas)
        model.assert_not_called()


def test_legacy_protocol_cannot_enable_parallel_mimo_before_model():
    with patch('subnet.miner.make_runtime') as model:
        with pytest.raises(ValueError,match='MiMo complete-group'):
            Miner(None,{},None,mimo_replicas=2)
        model.assert_not_called()


@pytest.mark.parametrize('replicas',[1,2,4])
def test_local_option_wraps_once_exact_n_and_closes_before_journal(tmp_path,replicas):
    m=mimo_manifest();m['deadline']=int(time.time())+60
    identity=Identity();bucket=FakeBucket()
    m['capabilities']={identity.id:capability(bucket,m,identity.id,registered_miners={identity.id})}
    with patch('subnet.model.check_runtime_profile'):
        actor=Miner(identity,m,'fixture',state_path=tmp_path/'state.json',mimo_replicas=replicas,
                    complete_group_transport=CompleteGroupTransport(http=FakeHTTP(bucket)))
    runtime=MimoCollectionRuntime(m,identity.id)
    closes=[]
    def close():
        assert actor._group_journal is not None
        closes.append(True)
    runtime.close=close
    with patch('subnet.miner.make_runtime',return_value=runtime),\
         patch('subnet.mimo_parallel_mining.wrap_runtime',return_value=runtime) as wrap:
        batch=actor.search(0,max_attempts=32)
        assert [r['seed']for r in batch['rollouts']]==list(range(8))
        assert actor.search(0)==batch
        if replicas==1:wrap.assert_not_called()
        else:wrap.assert_called_once_with(runtime,actor=actor,replicas=replicas)
        actor.close()
    assert len(closes)==1
    assert actor._group_journal is None and actor._closed




def test_backend_dispatch_preserves_local_replica_option_without_mutating_job():
    from subnet.backend_jobs import execute
    envelope={'payload':{'role':'mine'},'fixture_only':True}
    with patch('subnet.backend_jobs._validate',return_value=({'role':'mine'},{})),\
         patch('subnet.mimo_worker.selected',return_value=True),\
         patch('subnet.mimo_worker.execute',return_value={'fixture_only':'dispatched'}) as delegated:
        assert execute(envelope,'authority','workspace',mimo_replicas=4)=={'fixture_only':'dispatched'}
    assert delegated.call_args.kwargs['mimo_replicas']==4
    assert envelope=={'payload':{'role':'mine'},'fixture_only':True}


def test_parallel_admission_failure_keeps_original_error_and_closes_journal(tmp_path):
    m=mimo_manifest();m['deadline']=int(time.time())+60
    identity=Identity();bucket=FakeBucket()
    m['capabilities']={identity.id:capability(bucket,m,identity.id,registered_miners={identity.id})}
    with patch('subnet.model.check_runtime_profile'):
        actor=Miner(identity,m,'fixture',state_path=tmp_path/'state.json',mimo_replicas=4,
                    complete_group_transport=CompleteGroupTransport(http=FakeHTTP(bucket)))
    runtime=MimoCollectionRuntime(m,identity.id)
    assert not hasattr(runtime,'close')
    with patch('subnet.miner.make_runtime',return_value=runtime),\
         patch('subnet.mimo_parallel_mining.wrap_runtime',side_effect=ValueError('original admission failure')):
        with pytest.raises(ValueError,match='original admission failure'):
            try:
                actor.search(0)
            finally:
                actor.close()
    assert runtime.calls==[] and actor._group_journal is None and actor._closed
    actor.close()
