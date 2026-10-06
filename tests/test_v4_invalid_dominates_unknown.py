import copy,unittest
from unittest.mock import patch
import torch
import test_forced_sampling as tiny
import test_compact_threeway_sampling as compact
from subnet import fast_prefill_audit as fast
from subnet.audit_policy import InvalidSample
from subnet.batches import pack
from subnet.backend_jobs import audit
from subnet.harness import source_hash

class PairSession(tiny.Session):
 good_text=None
 def step(self,action):
  r=super().step(action);r['reward']=float(action['text']==self.good_text);r['classification']='positive'if r['reward']else'negative';return r

class InvalidDominates(unittest.TestCase):
 def setUp(self):
  self.fixture=compact.CompactThreeway();self.r,self.m=self.fixture.runtime();draws=[self.fixture.make(self.r,s)for s in range(16)]
  self.pos=draws[0];self.neg=next(v for v in draws if v[0]['turns'][0]['text']!=self.pos[0]['turns'][0]['text'])
  PairSession.good_text=self.pos[0]['turns'][0]['text']
  for rollout,arrays in (self.pos,self.neg):
   result=PairSession().step({'text':rollout['turns'][0]['text']})
   for key in ('reward','classification'):rollout[key]=rollout['turns'][0][key]=result[key]
  self.m.update(harness_source_hash=source_hash(),K=1,L=1,max_batches=1,environments=[dict(env_id='tiny',spec={'id':'tiny','version':'v1'},indices=[2],harness=self.r.harness)])
 def unknown(self):return fast._threeway_unknown('bounded uncertainty',torch.tensor([True]))
 def run_batch(self,second=None,effects=None):
  p,pa=copy.deepcopy(self.pos);n,na=copy.deepcopy(second or self.neg);batch=dict(schema=2,epoch=self.m['epoch'],checkpoint=self.m['checkpoint']['id'],env_id='tiny',environment_version='v1',index=2,sample_index=2,rollouts=[p,n])
  with patch.object(self.r,'for_environment',return_value=self.r),patch('subnet.model.create_session',side_effect=lambda spec:PairSession()),patch.object(fast,'verify_sampling',side_effect=effects or[self.unknown(),None]),patch.object(fast,'verify_cached_reference',side_effect=AssertionError('no replay')):
   return audit(pack([(batch,[pa,na])]),self.m,self.r)
 def test_unknown_first_cannot_hide_fresh_toploc_offpolicy_second(self):
  report,pairs=self.run_batch(effects=[self.unknown(),InvalidSample('CDF interval outside')]);o=report['outcomes'][0]
  self.assertFalse(o['valid']);self.assertEqual(o['failure_kind'],'confirmed_invalid');self.assertEqual(pairs,[])
 def test_real_fresh_proof_forgery_after_unknown_is_confirmed_invalid(self):
  from subnet import probability_artifacts as artifacts
  bad=copy.deepcopy(self.neg);turn=bad[0]['turns'][0];_,old_probs=self.r.compute(turn['prompt'],turn['output']);turn['output'][0]=next(int(t)for t in old_probs[0].argsort().tolist()if int(t)not in(turn['output'][0],self.r.tokenizer.eos_token_id));turn['text']=self.r.tokenizer.decode(turn['output']);acts,probs=self.r.compute(turn['prompt'],turn['output']);turn['proofs']=self.r.build_proofs(acts,decode_batching_size=16,topk=128);bad=(bad[0],[artifacts.encode(probs,turn['output'],self.m['probability_artifact_policy'])])
  result=PairSession().step({'text':turn['text']})
  for key in ('reward','classification'):bad[0][key]=turn[key]=result[key]
  original=fast.verify_sampling;calls=[]
  def first_unknown_then_actual(*args):
   calls.append(1)
   if len(calls)==1:raise self.unknown()
   return original(*args)
  report,pairs=self.run_batch(second=bad,effects=first_unknown_then_actual);self.assertEqual(len(calls),2);o=report['outcomes'][0];self.assertFalse(o['valid']);self.assertEqual(o['failure_kind'],'confirmed_invalid');self.assertIn('CDF interval outside',o['reason']);self.assertEqual(pairs,[])
 def test_early_unknown_cannot_hide_invalid_later_turn(self):
  class TwoTurns(PairSession):
   def __init__(self):self.calls=0
   def step(self,action):
    self.calls+=1;r=super().step(action);r['done']=self.calls==2;return r
  self.r.spec.max_turns=2
  with patch('subnet.model.create_session',side_effect=lambda spec:TwoTurns()):rollout,arrays=self.r.rollout(2,0)
  rollout['turns'][1]['text']='forged native text'
  with patch('subnet.model.create_session',side_effect=lambda spec:TwoTurns()),patch.object(fast,'verify_sampling',side_effect=self.unknown()):
   with self.assertRaisesRegex(InvalidSample,'text'):self.r.verify(rollout,arrays)
 def test_unknown_cannot_hide_forged_native_outcome_same_rollout(self):
  bad=copy.deepcopy(self.pos[0]);bad['turns'][0]['reward']=1-bad['turns'][0]['reward']
  with patch('subnet.model.create_session',return_value=PairSession()),patch.object(fast,'verify_sampling',side_effect=self.unknown()):
   with self.assertRaisesRegex(InvalidSample,'environment replay'):self.r.verify(bad,self.pos[1])
 def test_unknown_first_cannot_hide_native_false_second(self):
  bad=copy.deepcopy(self.neg);bad[0]['turns'][0]['classification']='positive'
  report,pairs=self.run_batch(second=bad);self.assertFalse(report['outcomes'][0]['valid']);self.assertEqual(report['outcomes'][0]['failure_kind'],'confirmed_invalid');self.assertEqual(pairs,[])
 def test_unknown_cannot_hide_native_verified_quota_failure(self):
  self.m.update(K=2,L=0);report,pairs=self.run_batch();o=report['outcomes'][0];self.assertFalse(o['valid']);self.assertEqual(o['failure_kind'],'confirmed_invalid');self.assertIn('quota',o['reason']);self.assertEqual(pairs,[])
 def test_honest_unknown_checks_both_native_outcomes_never_credit(self):
  report,pairs=self.run_batch();o=report['outcomes'][0];self.assertIsNone(o['valid']);self.assertEqual(o['failure_kind'],'numerical_ambiguous');self.assertTrue(o['environment_verification_complete']);self.assertFalse(o['sampling_verification_complete']);self.assertEqual(o['uncertain_rollouts'][0]['rollout'],0);self.assertEqual(pairs,[]);self.assertEqual(report['accepted'],[])
 def test_legacy_v3_still_raises_unknown_before_native_step(self):
  self.r.sampling_context['contract']['version']=fast.SUPPORT_VERSION
  bad=copy.deepcopy(self.pos[0]);bad['sampling']=self.r.sampling_receipt(bad['seed'])['sampling'];bad['turns'][0]['reward']=1-bad['turns'][0]['reward']
  with patch('subnet.model.create_session',return_value=PairSession()),patch.object(fast,'verify_sampling',side_effect=self.unknown()):
   with self.assertRaises(fast.NumericalAmbiguity):self.r.verify(bad,self.pos[1])
if __name__=='__main__':unittest.main()
