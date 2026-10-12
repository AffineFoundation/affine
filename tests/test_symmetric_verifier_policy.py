from pathlib import Path
import sys,copy,hashlib,json,math,types,unittest
from unittest.mock import patch
D=Path(__file__).resolve().parents[1];sys.path[:0]=[str(D),str(D/'tests')]
import torch
from subnet import forced_sampling as sampler,fast_prefill_audit as fast,symmetric_verifier_policy as policy
from subnet.audit_policy import InvalidSample
import test_fast_prefill_audit as fixture

class CandidateTests(unittest.TestCase):
 def runtime(self):
  r,m=fixture.Controls().support_runtime();m.update(K=4,L=4,max_batches=9,source_bundle={'sha256':'e'*64});m['sampling_contract'].update(version=sampler.MINER_VERSION,max_attempts=1000);m[policy.FIELD]=policy.current_policy();sampler.bind_runtime(r,m,'d'*64);r.sampling_verification_events=[];return r,m
 def test_prescribed_draws_and_receipts_are_identical_policy_off_on(self):
  r,m=self.runtime();a=copy.deepcopy(m);a.pop(policy.FIELD);before=sampler.binding(a,'d'*64);after=sampler.binding(m,'d'*64);self.assertEqual(before,after)
  for nonce in [0,1,999]:
   self.assertEqual(sampler.receipt(before,nonce),sampler.receipt(after,nonce))
   for position in [0,16,2047]:self.assertEqual(sampler.uniform(before,'math','c'*64,2,nonce,0,position),sampler.uniform(after,'math','c'*64,2,nonce,0,position))
 def test_bad_nonce(self):
  r,m=self.runtime()
  for n in [-1,1000,True,1.0]:
   with self.subTest(n=n),self.assertRaises(ValueError):sampler.receipt(r.sampling_context,n)
 def test_source_bound_policy(self):
  r,m=self.runtime();bad=copy.deepcopy(m);bad[policy.FIELD]['source_files']['subnet/fast_prefill_audit.py']='0'*64
  with self.assertRaises(ValueError):fast.bind(bad,r.harness)
 def test_policy_requires_source_bundle(self):
  r,m=self.runtime();m.pop('source_bundle')
  with self.assertRaises(ValueError):policy.for_manifest(m)
 def test_wrong_sampler_source_rejected(self):
  r,m=self.runtime();m['sampling_source_hash']='0'*64
  with self.assertRaises(ValueError):sampler.binding(m,'d'*64)
 def test_inside_boundary_old_accept_new_escalates(self):
  x=torch.log(torch.tensor([[.5,.5]]));self.assertTrue(fast.verify_intervals(x,[0],[.499999],1.,1.,1e-5)['all_intervals_verified'])
  with self.assertRaises(fast.NumericalAmbiguity):policy.intervals(x,[0],[.499999],1.,1.,1e-5)
 def test_outside_boundary_escalates(self):
  with self.assertRaises(fast.NumericalAmbiguity):policy.intervals(torch.log(torch.tensor([[.5,.5]])),[0],[.500001],1.,1.,1e-5)
 def test_far_outside_rejects(self):
  with self.assertRaises(InvalidSample):policy.intervals(torch.log(torch.tensor([[.5,.5]])),[0],[.8],1.,1.,1e-5)
 def test_interior_passes(self):self.assertTrue(policy.intervals(torch.log(torch.tensor([[.5,.5]])),[0],[.2],1.,1.,1e-5)['all_intervals_verified'])
 def test_support_exclusion_requires_exact(self):
  with self.assertRaises(fast.NumericalAmbiguity):policy.intervals(torch.log(torch.tensor([[.999999,.000001]])),[1],[.999999],1.,.9,1e-3)
 def test_nonfinite_model_cdf_is_not_invalid_sample(self):
  with self.assertRaises(RuntimeError)as ctx:policy.intervals(torch.tensor([[float('nan'),0.]]),[0],[.2],1.,1.,1e-5)
  self.assertNotIsInstance(ctx.exception,InvalidSample)
 def test_nonfinite_draw_is_operator_error(self):
  with self.assertRaises(ValueError)as ctx:policy.intervals(torch.tensor([[0.,0.]]),[0],[float('nan')],1.,1.,1e-5)
  self.assertNotIsInstance(ctx.exception,InvalidSample)
 def test_actual_cached_reference_matches(self):
  r,m=self.runtime();out=fast.cached_sample(r,[0,1],0,0,2,'c'*64);roll=dict(seed=0,index=2,task_hash='c'*64)
  with patch.object(policy,'intervals',side_effect=fast.NumericalAmbiguity('boundary')):result=fast.verify_sampling(r,roll,0,[0,1],out,torch.zeros((len(out),7)))
  self.assertEqual(result['sampling_adjudication'],'exact-cached-match');self.assertEqual(r.sampling_verification_events[-1]['adjudication'],'exact-cached-match')
 def test_actual_cached_reference_rejects_token_mutation(self):
  r,m=self.runtime();out=fast.cached_sample(r,[0,1],0,0,2,'c'*64);out[0]=(out[0]+1)%6;roll=dict(seed=0,index=2,task_hash='c'*64)
  with patch.object(policy,'intervals',side_effect=fast.NumericalAmbiguity('boundary')),self.assertRaises(InvalidSample):fast.verify_sampling(r,roll,0,[0,1],out,torch.zeros((len(out),7)))
  self.assertEqual(r.sampling_verification_events[-1]['adjudication'],'exact-cached-mismatch')
 def test_cached_unavailable_unknown(self):
  r,m=self.runtime();out=fast.cached_sample(r,[0,1],0,0,2,'c'*64);roll=dict(seed=0,index=2,task_hash='c'*64)
  with patch.object(policy,'intervals',side_effect=fast.NumericalAmbiguity('boundary')),patch.object(fast,'verify_cached_reference',side_effect=RuntimeError('infra')),self.assertRaises(fast.NumericalAmbiguity):fast.verify_sampling(r,roll,0,[0,1],out,torch.zeros((len(out),7)))
  self.assertEqual(r.sampling_verification_events[-1]['adjudication'],'exact-cached-unavailable')
 def test_wrong_miner_context(self):
  r,m=self.runtime();r.sampling_context['checkpoint']='0'*64
  with self.assertRaises(ValueError):fast.verify_sampling(r,dict(seed=0,index=2,task_hash='c'*64),0,[0,1],[6],torch.zeros((1,7)))
 def test_warm_rebind_policy_off_uses_original(self):
  r,m=self.runtime();old=copy.deepcopy(m);old.pop(policy.FIELD);sampler.bind_runtime(r,old,'d'*64)
  with patch.object(policy,'verify_sampling',side_effect=AssertionError('policy leaked')),patch.object(fast,'verify_intervals',return_value={'all_intervals_verified':True}):self.assertTrue(fast.verify_sampling(r,dict(seed=0,index=2,task_hash='c'*64),0,[0,1],[r.tokenizer.eos_token_id],torch.zeros((1,7)))['all_intervals_verified'])
 def report(self):
  r,m=self.runtime();roll={'turns':[{'output':[1,6]}]};event=dict(rollout_sha256=policy.digest(roll),turn=0,positions=2,adjudication='calibrated-interior',seconds=.1);rep=dict(accepted=[{'env_id':'math','index':0,'rollouts':[roll]}],outcomes=[dict(batch=0,env_id='math',index=0,valid=True,fully_audited=True,sampling_verification_events=[event])],sampling_verification_policy=m[policy.FIELD]);return m,rep
 def test_accepted_report_requires_exact_events(self):m,r=self.report();policy.require_report(m,r)
 def test_missing_policy_cannot_credit(self):
  m,r=self.report();r.pop(policy.FIELD)
  with self.assertRaises(ValueError):policy.require_report(m,r)
 def test_missing_event_cannot_credit(self):
  m,r=self.report();r['outcomes'][0]['sampling_verification_events']=[]
  with self.assertRaises(ValueError):policy.require_report(m,r)
 def test_duplicate_event_cannot_credit(self):
  m,r=self.report();r['outcomes'][0]['sampling_verification_events']*=2
  with self.assertRaises(ValueError):policy.require_report(m,r)
 def test_unavailable_cannot_be_fraud(self):
  m,r=self.report();r['accepted']=[];o=r['outcomes'][0];o.update(valid=False,fully_audited=True,failure_kind='confirmed_invalid');o['sampling_verification_events'][0]['adjudication']='exact-cached-unavailable'
  with self.assertRaises(ValueError):policy.require_report(m,r)
 def test_unavailable_can_be_unknown(self):
  m,r=self.report();r['accepted']=[];o=r['outcomes'][0];o.update(valid=None,fully_audited=False,failure_kind='numerical_ambiguous');o['sampling_verification_events'][0]['adjudication']='exact-cached-unavailable';policy.require_report(m,r)
 def test_exact_mismatch_cannot_credit(self):
  m,r=self.report();r['outcomes'][0]['sampling_verification_events'][0]['adjudication']='exact-cached-mismatch'
  with self.assertRaises(ValueError):policy.require_report(m,r)
 def test_structural_rejection_needs_no_model_event(self):
  m,r=self.report();r=dict(accepted=[],outcomes=[dict(valid=False,fully_audited=False,failure_kind='structural_invalid')]);policy.require_report(m,r)
class SuccessorReportTests(unittest.TestCase):
 runtime=CandidateTests.runtime
 report=CandidateTests.report
 def test_fully_valid_empty_accepted_refused(self):
  m,r=self.report();r['accepted']=[];r['outcomes'][0]['sampling_verification_events']=[]
  with self.assertRaises(ValueError):policy.require_report(m,r)
 def test_valid_without_fully_audited_refused(self):
  m,r=self.report();r['accepted']=[];r['outcomes'][0]['fully_audited']=False
  with self.assertRaises(ValueError):policy.require_report(m,r)
 def test_duplicate_accepted_refused(self):
  m,r=self.report();r['accepted']*=2
  with self.assertRaises(ValueError):policy.require_report(m,r)
 def test_duplicate_valid_outcome_refused(self):
  m,r=self.report();r['outcomes']*=2
  with self.assertRaises(ValueError):policy.require_report(m,r)
 def test_outcome_wrong_task_refused(self):
  m,r=self.report();r['outcomes'][0]['index']=1
  with self.assertRaises(ValueError):policy.require_report(m,r)
 def test_cross_child_event_swaps_refused(self):
  m,r=self.report();b=copy.deepcopy(r['accepted'][0]);b['index']=1;b['rollouts'][0]['turns'][0]['output']=[2,6];o=copy.deepcopy(r['outcomes'][0]);o.update(batch=1,index=1);o['sampling_verification_events'][0]['rollout_sha256']=policy.digest(b['rollouts'][0]);r['accepted'].append(b);r['outcomes'].append(o);policy.require_report(m,r)
  r['outcomes'][0]['sampling_verification_events'],o['sampling_verification_events']=o['sampling_verification_events'],r['outcomes'][0]['sampling_verification_events']
  with self.assertRaises(ValueError):policy.require_report(m,r)
 def test_transport_label_cannot_hide_valid_outcome(self):
  m,r=self.report();r.pop(policy.FIELD);r['accepted']=[];r['outcomes'][0].update(rejection_stage='transport',fully_audited=False)
  with self.assertRaises(ValueError):policy.require_report(m,r)
 def test_stop_framing_event_and_existing_invalid_type(self):
  r,m=self.runtime();roll=dict(seed=0,index=2,task_hash='c'*64)
  with self.assertRaises(InvalidSample):fast.verify_sampling(r,roll,0,[0,1],[r.tokenizer.eos_token_id,0],torch.zeros((2,7)))
  self.assertEqual(r.sampling_verification_events[-1]['adjudication'],'invalid-stop-framing')
if __name__=='__main__':unittest.main()
