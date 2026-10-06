import copy,types,unittest
from unittest.mock import patch
import torch
import test_forced_sampling as tiny
import test_fast_prefill_audit as prefill
from subnet import fast_prefill_audit as fast,forced_sampling as f,probability_artifacts as artifacts
from subnet.audit_policy import InvalidSample
from subnet.batches import pack

class CompactThreeway(unittest.TestCase):
 def runtime(self,large=False):
  r,m=prefill.Controls().runtime()
  if large:
   with torch.random.fork_rng():
    torch.manual_seed(81);r.model.embedding=torch.nn.Embedding(128,128);r.model.head=torch.nn.Linear(128,128)
   r.model.config.vocab_size=128;r.tokenizer.eos_token_id=127
  m['sampling_contract']=f.new_contract(dict(version=fast.THREEWAY_VERSION,max_attempts=16,calibration=m['sampling_contract']['calibration'],uncertainty_adjudication='numerical-inconclusive-no-replay-v1'))
  m['sampling_contract']['randomness']='b'*64;m['probability_artifact_policy']={'version':artifacts.VERSION}
  return artifacts.bind_runtime(f.bind_runtime(r,m),m),m
 def make(self,r,seed=0):
  with patch('subnet.model.create_session',return_value=tiny.Session()):return r.rollout(2,seed)
 def verify(self,r,rollout,arrays):
  with patch('subnet.model.create_session',return_value=tiny.Session()),patch.object(fast,'verify_cached_reference',side_effect=AssertionError('threeway must never replay')):return r.verify(rollout,arrays)
 def test_honest_compact_toploc_one_prefill_and_legacy_policy_unchanged(self):
  r,m=self.runtime();rollout,arrays=self.make(r);before=r.model.calls
  self.assertTrue(self.verify(r,rollout,arrays));self.assertEqual(r.model.calls-before,1)
  self.assertEqual(arrays[0].shape,(len(rollout['turns'][0]['output']),1));self.assertTrue(rollout['turns'][0]['proofs'])
  old,_=prefill.Controls().support_runtime();self.assertEqual(old.sampling_context['contract']['version'],fast.SUPPORT_VERSION)
 def test_eighty_distinct_fresh_toploc_offpolicy_swaps_rejected(self):
  r,m=self.runtime(large=True);original,arrays=self.make(r);turn=original['turns'][0];_,lp=r.compute(turn['prompt'],turn['output'])
  choices=[int(x)for x in torch.as_tensor(lp[0]).argsort().tolist()if int(x)not in (turn['output'][0],r.tokenizer.eos_token_id)][:80]
  self.assertEqual(len(set(choices)),80)
  for token in choices:
   bad=copy.deepcopy(original);t=bad['turns'][0];t['output'][0]=token;t['text']=r.tokenizer.decode(t['output']);acts,probs=r.compute(t['prompt'],t['output']);t['proofs']=r.build_proofs(acts,decode_batching_size=16,topk=128)
   # Verify genuine new TOPLOC evidence independently; failure must be CDF, not stale proof.
   self.assertTrue(all(x.exp_mismatches==0 for x in r.verify_proofs(acts,t['proofs'],decode_batching_size=16,topk=128)))
   a=[artifacts.encode(probs,t['output'],m['probability_artifact_policy'])];out=tiny.Session().step({'text':t['text']})
   for key in ('reward','classification'):t[key]=bad[key]=out[key]
   with self.subTest(token=token),self.assertRaisesRegex(InvalidSample,'CDF interval outside'):self.verify(r,bad,a)
 def test_uploaded_selected_probs_and_toploc_are_still_required(self):
  r,m=self.runtime();rollout,arrays=self.make(r)
  bad=[arrays[0].copy()];bad[0][0,0]-=.1
  with self.assertRaisesRegex(InvalidSample,'probabilities'):self.verify(r,rollout,bad)
  for proofs in ([],['invalid-base64']):
   bad=copy.deepcopy(rollout);bad['turns'][0]['proofs']=proofs
   with self.assertRaises(InvalidSample):self.verify(r,bad,arrays)
 def test_epoch_checkpoint_seed_and_receipt_rebinding_rejected(self):
  r,m=self.runtime();rollout,arrays=self.make(r)
  for field in ('epoch','checkpoint'):
   changed=copy.deepcopy(m);changed[field]='different'if field=='epoch'else{'id':'d'*64};other,_=self.runtime()
   if field=='checkpoint':
    with self.assertRaises(fast.CalibrationRequired):f.bind_runtime(other,changed)
   else:
    f.bind_runtime(other,changed)
    with self.assertRaises(InvalidSample):self.verify(other,rollout,arrays)
  different=next(seed for seed in range(1,16)if self.make(r,seed)[0]['turns'][0]['output']!=rollout['turns'][0]['output'])
  rebound=copy.deepcopy(rollout);rebound['seed']=different;rebound['sampling']=f.receipt(r.sampling_context,different)
  with self.assertRaises(InvalidSample):self.verify(r,rebound,arrays)
  bad=copy.deepcopy(rollout);bad['seed']=128
  with self.assertRaises(InvalidSample):self.verify(r,bad,arrays)
  bad=copy.deepcopy(rollout);bad['sampling']['binding_sha256']='0'*64
  with self.assertRaises(InvalidSample):self.verify(r,bad,arrays)
 def test_v4_requires_exact_explicit_signed_fields(self):
  r,m=self.runtime();c=m['sampling_contract'];self.assertEqual(f.validate(c)['verification'],'prefill-cdf-calibrated-threeway')
  for key,val in [('uncertainty_adjudication',None),('uncertainty_adjudication',False),('verification','prefill-cdf-calibrated'),('generation','uncached-eager-inverse-cdf')]:
   bad=copy.deepcopy(c);bad[key]=val
   with self.assertRaises(ValueError):f.validate(bad)
  bad=copy.deepcopy(c);bad.pop('uncertainty_adjudication')
  with self.assertRaises(ValueError):f.validate(bad)
  bad=copy.deepcopy(c);bad['version']=fast.SUPPORT_VERSION
  with self.assertRaises(ValueError):f.validate(bad)
 def test_both_boundary_sides_unknown_outside_invalid_zero_support_never_valid(self):
  lp=torch.tensor([[.5,.5]]).log()
  for token,u in [(0,.4995),(0,.5005),(1,.4995),(1,.5005)]:
   with self.assertRaises(fast.NumericalAmbiguity)as ctx:fast.verify_threeway_intervals(lp,[token],[u],1.,1.,.001)
   self.assertEqual(ctx.exception.uncertain_positions,[0]);self.assertEqual(ctx.exception.uncertain_position_count,1)
  with self.assertRaises(InvalidSample):fast.verify_threeway_intervals(lp,[0],[.8],1.,1.,.001)
  zero=torch.tensor([[.9,.1]]).log()
  with self.assertRaises(fast.NumericalAmbiguity):fast.verify_threeway_intervals(zero,[1],[.9995],1.,.8,.001)
  with self.assertRaises(InvalidSample):fast.verify_threeway_intervals(zero,[1],[.5],1.,.8,.001)
 def test_successor_hook_accepts_v4_only_with_explicit_full_draw_contract(self):
  from subnet import successor_calibration as c
  from subnet.harness import normalize
  r,m=self.runtime();h={**r.harness,'max_output_tokens':8};cal={**m['sampling_contract']['calibration'],'harness_sha256':fast.digest(normalize(h))}
  d=f.new_contract(dict(version=fast.THREEWAY_VERSION,max_attempts=16,calibration=cal,uncertainty_adjudication='numerical-inconclusive-no-replay-v1'))
  req=dict(version=c.VERSION,env_id='tiny',harness=h,task_indices=[2,3],max_tokens=8,draw_contract=d)
  self.assertEqual(c.request(req)['draw_contract'],d)
  bad=copy.deepcopy(req);bad['draw_contract'].pop('uncertainty_adjudication')
  with self.assertRaises(ValueError):c.request(bad)
 def test_actual_audit_report_unknown_is_signed_not_accepted_or_fraud(self):
  from subnet.backend_jobs import audit
  from subnet.storage import canonical
  from nacl.signing import SigningKey
  r,m=self.runtime();rollout,arrays=self.make(r);second,second_arrays=self.make(r,1)
  from subnet.harness import source_hash
  m.update(harness_source_hash=source_hash(),K=1,L=1,max_batches=1,environments=[dict(env_id='tiny',spec={'id':'tiny','version':'v1'},indices=[2],harness=r.harness)])
  batch=dict(schema=2,epoch=m['epoch'],checkpoint=m['checkpoint']['id'],env_id='tiny',environment_version='v1',index=2,sample_index=2,rollouts=[rollout,second]);data=pack([(batch,[arrays,second_arrays])])
  with patch.object(r,'for_environment',return_value=r),patch('subnet.model.create_session',return_value=tiny.Session()),patch.object(fast,'verify_threeway_intervals',side_effect=fast._threeway_unknown('bounded numeric uncertainty',torch.tensor([True]))),patch.object(fast,'verify_cached_reference',side_effect=AssertionError('forbidden replay')):
   report,pairs=audit(data,m,r)
  self.assertEqual(report['accepted'],[]);self.assertEqual(pairs,[]);o=report['outcomes'][0];self.assertIsNone(o['valid']);self.assertFalse(o['fully_audited']);self.assertEqual(o['failure_kind'],'numerical_ambiguous');self.assertFalse(o['sampling_verification_complete']);self.assertFalse(o['environment_verification_complete']);self.assertEqual(o['uncertain_token_positions'],[0]);self.assertEqual(o['uncertain_token_position_count'],1)
  key=SigningKey.generate();sig=key.sign(canonical(report)).signature;key.verify_key.verify(canonical(report),sig)
  changed=copy.deepcopy(report);changed['outcomes'][0]['valid']=True
  with self.assertRaises(Exception):key.verify_key.verify(canonical(changed),sig)
