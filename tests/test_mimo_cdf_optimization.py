import importlib.util
import math
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from subnet.complete_group_distribution import distribution, verify_intervals as original_intervals
from subnet.audit_policy import InvalidSample
from subnet.fast_prefill_audit import NumericalAmbiguity, SupportMismatch

from subnet import complete_group_distribution as candidate
original_intervals=candidate.verify_intervals_reference

@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)

@pytest.mark.parametrize('vocab,k',[(31,1),(31,20),(31,0),(31,31),(1024,20),(248320,20)])
@pytest.mark.parametrize('top_p',[.95,1.])
def test_exact_probability_and_cdf_parity(vocab,k,top_p):
    gen=torch.Generator().manual_seed(719)
    rows=torch.randn((7,vocab),generator=gen)
    rows[0]=0
    rows[1]=rows[1].round()  # repeated values around the top-k cutoff
    rows[2]=rows[2]*100  # underflow within retained support
    for temperature in (.6,1.,2.):
        old=distribution(rows,temperature,top_p,k)
        new=candidate.audit_distribution(rows,temperature,top_p,k)
        assert torch.equal(old,new)
        assert torch.equal(old.cumsum(-1),new.cumsum(-1))
        assert torch.equal(old[3],candidate.audit_distribution(rows[3],temperature,top_p,k))


def result(function,*args,**kw):
    try:return ('ok',function(*args,**kw))
    except (InvalidSample,NumericalAmbiguity,RuntimeError,ValueError) as e:return (type(e).__name__,str(e))


def test_all_positions_and_chunk_boundaries_match_reference():
    rows=torch.randn((70,1024),generator=torch.Generator().manual_seed(91))
    config={'temperature':1.,'top_p':.95,'top_k':20}
    p=distribution(rows,**config);ids=p.argmax(-1).tolist()
    cdf=p.cumsum(-1);idx=torch.arange(len(rows));tok=torch.tensor(ids)
    mid=(cdf[idx,tok]-p[idx,tok]/2).tolist()
    for chunk in (1,2,7,32,128):
        assert candidate.verify_intervals(rows,ids,mid,config,0.,row_chunk_size=chunk)==original_intervals(rows,ids,mid,config,0.)
    for pos in (0,31,32,69):
        wrong=ids.copy();wrong[pos]=int(torch.argmin(p[pos]))
        for chunk in (1,32):
            assert result(candidate.verify_intervals,rows,wrong,mid,config,0.,row_chunk_size=chunk)==result(original_intervals,rows,wrong,mid,config,0.)


def test_boundary_error_precedence_and_nonfinite_late_chunk():
    rows=torch.tensor([[3.,2.,1.,0.]]*35)
    config={'temperature':1.,'top_p':1.,'top_k':2}
    p=distribution(rows,**config);ids=[0]*35;draws=[.1]*35
    # Earlier out-of-region, later excluded support: full reference prioritizes support.
    draws[0]=.999;ids[-1]=3
    assert result(candidate.verify_intervals,rows,ids,draws,config,1e-4)==result(original_intervals,rows,ids,draws,config,1e-4)
    rows[-1,-1]=float('nan')
    assert result(candidate.verify_intervals,rows,ids,draws,config,1e-4)==result(original_intervals,rows,ids,draws,config,1e-4)
    rows=torch.tensor([[3.,2.,1.,0.]])
    edge=float(distribution(rows,**config)[0,0]);draws=[edge+1e-6]
    assert result(candidate.verify_intervals,rows,[0],draws,config,1e-4)==result(original_intervals,rows,[0],draws,config,1e-4)


def test_only_retained_support_is_sorted_and_input_unchanged():
    rows=torch.randn((3,248320),generator=torch.Generator().manual_seed(2));before=rows.clone()
    real_sort=torch.sort;calls=[]
    def bounded_sort(values,*args,**kw):
        calls.append(values.shape[-1]);assert values.shape[-1]<=20
        return real_sort(values,*args,**kw)
    with patch('torch.sort',side_effect=bounded_sort):
        candidate.audit_distribution(rows,1.,.95,20)
    assert calls==[20] and torch.equal(rows,before)


def test_exact_cached_fallback_still_rejects_mutated_token():
    from test_complete_group_mimo import mimo_manifest,toy_runtime
    from subnet.complete_group_sampling import cached_sample
    from subnet.complete_group_verification import sampling_check
    m=mimo_manifest();r=toy_runtime(m,'a'*64)
    tokens=cached_sample(r,[1,2],0,0,0,'b'*64)
    rows=torch.zeros((2,248320)) # Deliberately support-inconsistent prefill to force exact fallback.
    row={'seed':0,'index':0,'task_hash':'b'*64}
    with patch('subnet.complete_group_verification.verify_intervals',candidate.verify_intervals):
        assert sampling_check(r,row,[1,2],tokens,rows.numpy())['cached_reference_adjudication']
        changed=tokens.copy();changed[0]=(changed[0]+1)%24
        with pytest.raises(InvalidSample,match='cached group sampler mismatch'):
            sampling_check(r,row,[1,2],changed,rows.numpy())

@pytest.mark.parametrize('size',[False,0,-1,129,1.5])
def test_unbounded_chunks_reject(size):
    with pytest.raises(ValueError,match='chunk'):
        candidate.verify_intervals(torch.zeros((2,3)),[0,0],[0.,0.],{'temperature':1.,'top_p':1.,'top_k':1},0.,row_chunk_size=size)


def test_early_disagreement_only_skips_fast_work_and_reports_checked_prefix():
    rows=torch.tensor([[3.,2.,1.,0.]]*70)
    config={'temperature':1.,'top_p':1.,'top_k':2}
    ids=[0]*70;ids[0]=3
    with patch.object(candidate,'audit_distribution',wraps=candidate.audit_distribution) as scan:
        with pytest.raises(SupportMismatch) as failure:
            candidate.verify_intervals(rows,ids,[.1]*70,config,0.,exact_fallback_on_disagreement=True)
    assert scan.call_count==1 and failure.value.checked_positions==32
    # A later infrastructure fault must never be hidden by the earlier mismatch.
    rows[-1,-1]=float('nan')
    with patch.object(candidate,'audit_distribution',wraps=candidate.audit_distribution) as scan:
        with pytest.raises(RuntimeError,match='nonfinite'):
            candidate.verify_intervals(rows,ids,[.1]*70,config,0.,exact_fallback_on_disagreement=True)
    scan.assert_not_called()


def test_no_disagreement_still_checks_every_position_in_early_mode():
    rows=torch.tensor([[3.,2.,1.,0.]]*70)
    config={'temperature':1.,'top_p':1.,'top_k':2}
    with patch.object(candidate,'audit_distribution',wraps=candidate.audit_distribution) as scan:
        result=candidate.verify_intervals(rows,[0]*70,[.1]*70,config,0.,exact_fallback_on_disagreement=True)
    assert scan.call_count==3 and result['positions']==70 and result['all_intervals_verified']
