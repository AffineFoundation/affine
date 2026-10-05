import copy,types,unittest
from unittest.mock import patch
import torch
import test_forced_sampling as fixture
from subnet import forced_sampling as f
from subnet import fast_prefill_audit as fast
from subnet.audit_policy import InvalidSample
from subnet.harness import normalize
class CachedTiny(fixture.TinyModel):
 def __init__(self):super().__init__();self.calls=0
 def forward(self,ids,output_hidden_states=False,use_cache=False,past_key_values=None):
  self.calls+=1;hidden=self.embedding(ids).cumsum(1)
  if past_key_values is not None:hidden=hidden+past_key_values
  return types.SimpleNamespace(logits=self.head(hidden),hidden_states=[hidden],past_key_values=hidden[:,-1:].detach()if use_cache else None)
class Controls(unittest.TestCase):
 def runtime(self):
  case=fixture.ForcedSamplingTests();case.setUp();runtime=case.runtime();runtime.model=CachedTiny().eval();manifest=copy.deepcopy(case.manifest)
  manifest.update(model_runtime_revision='cpu-test-v1',backend_profile={'device':'cpu'})
  calibration=dict(version=fast.CALIBRATION,checkpoint='a'*64,model_runtime_revision='cpu-test-v1',backend_profile_sha256=fast.digest(manifest['backend_profile']),harness_sha256=fast.digest(normalize(runtime.harness)),report_sha256='c'*64,cdf_abs_error=1e-5,logprob_atol=1e-5,toploc_exp_mismatches=0,toploc_mant_err_mean=0,toploc_mant_err_median=0)
  manifest['sampling_contract']=f.new_contract(dict(version=fast.VERSION,max_attempts=16,calibration=calibration));manifest['sampling_contract']['randomness']='b'*64
  return f.bind_runtime(runtime,manifest),manifest
 def support_runtime(self):
  runtime,manifest=self.runtime();manifest['sampling_contract']=f.new_contract(dict(version=fast.SUPPORT_VERSION,max_attempts=16,calibration=manifest['sampling_contract']['calibration'],support_adjudication='exact-cached-replay-v1'));manifest['sampling_contract']['randomness']='b'*64
  return f.bind_runtime(runtime,manifest),manifest
 def test_support_option_requires_exact_signed_contract(self):
  runtime,m=self.support_runtime();self.assertEqual(f.validate(m['sampling_contract'])['support_adjudication'],'exact-cached-replay-v1')
  for change in ('missing','wrong'):
   wrong=copy.deepcopy(m['sampling_contract'])
   if change=='missing':wrong.pop('support_adjudication')
   else:wrong['support_adjudication']='accept-excluded-tokens'
   with self.assertRaises(ValueError):f.validate(wrong)
  wrong=copy.deepcopy(m['sampling_contract']);wrong['version']=fast.VERSION
  with self.assertRaises(ValueError):f.validate(wrong)
 def test_honest_support_shift_uses_actual_reference_and_forged_tail_fails(self):
  runtime,m=self.support_runtime();prompt=[0,1];task='c'*64;output=fast.cached_sample(runtime,prompt,0,0,2,task);rollout=dict(seed=0,index=2,task_hash=task)
  # Isolate a nucleus support crossing in reference probabilities. Actual
  # cached execution determines the verdict; no tolerance makes zero valid.
  runtime.harness={**runtime.harness,'top_p':.9};output=fast.cached_sample(runtime,prompt,0,0,2,task)
  probs=torch.full((len(output),7),-30.)
  for position,token in enumerate(output):probs[position,(token+1)%7]=0.
  before=runtime.model.calls;result=fast.verify_sampling(runtime,rollout,0,prompt,output,probs);self.assertTrue(result['cached_reference_adjudication']);self.assertEqual(runtime.model.calls-before,len(output))
  forged=list(output);forged[0]=(forged[0]+1)%6
  if forged[0]==output[0]:forged[0]=(forged[0]+1)%6
  for position,token in enumerate(forged):probs[position].fill_(-30.);probs[position,(token+1)%7]=0.
  with self.assertRaises(InvalidSample):fast.verify_sampling(runtime,rollout,0,prompt,forged,probs)
 def test_v3_boundary_ambiguity_requires_real_cached_reference(self):
  runtime,m=self.support_runtime();prompt=[0,1];task='c'*64;output=fast.cached_sample(runtime,prompt,0,0,2,task);rollout=dict(seed=0,index=2,task_hash=task)
  before=runtime.model.calls
  with patch.object(fast,'verify_intervals',side_effect=fast.NumericalAmbiguity('simulated boundary crossing')):
   result=fast.verify_sampling(runtime,rollout,0,prompt,output,torch.zeros((len(output),7)))
   self.assertTrue(result['cached_reference_adjudication']);self.assertEqual(runtime.model.calls-before,len(output))
   forged=list(output);forged[0]=(forged[0]+1)%6
   with self.assertRaises(InvalidSample):fast.verify_sampling(runtime,rollout,0,prompt,forged,torch.zeros((len(output),7)))
 def test_v3_reference_infrastructure_failure_remains_unknown(self):
  runtime,m=self.support_runtime();prompt=[0,1];task='c'*64;output=fast.cached_sample(runtime,prompt,0,0,2,task)
  with patch.object(fast,'verify_intervals',side_effect=fast.NumericalAmbiguity('boundary')),patch.object(fast,'verify_cached_reference',side_effect=RuntimeError('GPU unavailable')):
   with self.assertRaises(fast.NumericalAmbiguity):fast.verify_sampling(runtime,dict(seed=0,index=2,task_hash=task),0,prompt,output,torch.zeros((len(output),7)))
 def test_old_v2_ambiguity_does_not_gain_reference_fallback(self):
  runtime,m=self.runtime();prompt=[0,1];task='c'*64;output=fast.cached_sample(runtime,prompt,0,0,2,task)
  with patch.object(fast,'verify_intervals',side_effect=fast.NumericalAmbiguity('boundary')),patch.object(fast,'verify_cached_reference',side_effect=AssertionError('old contract must not replay')):
   with self.assertRaises(fast.NumericalAmbiguity):fast.verify_sampling(runtime,dict(seed=0,index=2,task_hash=task),0,prompt,output,torch.zeros((len(output),7)))
 def test_old_fast_contract_support_exclusion_stays_invalid(self):
  runtime,m=self.runtime();runtime.harness={**runtime.harness,'top_p':.9};output=fast.cached_sample(runtime,[0,1],0,0,2,'c'*64);probs=torch.full((len(output),7),-30.)
  for position,token in enumerate(output):probs[position,(token+1)%7]=0.
  with self.assertRaises(fast.SupportMismatch):fast.verify_sampling(runtime,dict(seed=0,index=2,task_hash='c'*64),0,[0,1],output,probs)
 def test_real_cached_generation_toploc_and_one_prefill_verification(self):
  miner,m=self.runtime();verifier,_=self.runtime();verifier.sampling_context=miner.sampling_context
  with patch('subnet.model.create_session',return_value=fixture.Session()):
   rollout,arrays=miner.rollout(2,0);before=verifier.model.calls
   with patch.object(verifier,'sample_output',side_effect=AssertionError('no autoregressive replay')):self.assertTrue(verifier.verify(rollout,arrays))
   self.assertEqual(verifier.model.calls-before,len(rollout['turns']))
 def test_synthesized_tokens_and_fresh_genuine_toploc_still_rejected(self):
  miner,m=self.runtime()
  with patch('subnet.model.create_session',return_value=fixture.Session()):
   rollout,arrays=miner.rollout(2,0);turn=rollout['turns'][0];turn['output'][0]=(turn['output'][0]+1)%7;turn['text']=miner.tokenizer.decode(turn['output']);acts,arrays[0]=miner.compute(turn['prompt'],turn['output']);turn['proofs']=miner.build_proofs(acts,decode_batching_size=16,topk=128);result=fixture.Session().step({'text':turn['text']})
   for key in ('reward','classification'):turn[key]=rollout[key]=result[key]
   with self.assertRaises(InvalidSample):miner.verify(rollout,arrays)
 def test_near_boundary_is_unknown_not_fraud_or_success(self):
  logits=torch.log(torch.tensor([[.5,.5]]))
  with self.assertRaises(fast.NumericalAmbiguity):fast.verify_intervals(logits,[0],[.500001],1.,1.,1e-5)
  with self.assertRaises(InvalidSample):fast.verify_intervals(logits,[0],[.8],1.,1.,1e-5)
  self.assertTrue(fast.verify_intervals(logits,[0],[.2],1.,1.,1e-5)['all_intervals_verified'])
 def test_nucleus_excluded_token_never_legalized_by_tolerance(self):
  probs=torch.log(torch.tensor([[.999999,.000001]]))
  with self.assertRaises(InvalidSample):fast.verify_intervals(probs,[1],[.999999],1.,.9,1e-3)
 def test_right_boundary_strict_and_bad_thresholds(self):
  logits=torch.log(torch.tensor([[.5,.5]]))
  with self.assertRaises(InvalidSample):fast.verify_intervals(logits,[0],[.5],1.,1.,0)
  self.assertTrue(fast.verify_intervals(logits,[1],[.5],1.,1.,0)['all_intervals_verified'])
  for value in (float('nan'),.1,-1,True):
   with self.assertRaises(ValueError):fast.verify_intervals(logits,[0],[.2],1.,1.,value)
 def test_calibration_cannot_move_to_other_checkpoint_profile_harness(self):
  runtime,m=self.runtime()
  for field in ('checkpoint','backend_profile','model_runtime_revision'):
   bad=copy.deepcopy(m);bad[field]={'id':'f'*64}if field=='checkpoint'else{'device':'foreign'}if field=='backend_profile'else'foreign'
   with self.assertRaises(fast.CalibrationRequired):fast.bind(bad,runtime.harness)
  with self.assertRaises(fast.CalibrationRequired):fast.bind(m,dict(runtime.harness,temperature=.8))
 def test_executed_cached_prefill_calibration_reports_real_forwards(self):
  runtime,m=self.runtime();before=runtime.model.calls
  evidence=fast.measure_cached_prefill(runtime,[0,1],checkpoint='a'*64,task_hash='c'*64,index=2,max_tokens=4)
  self.assertEqual(runtime.model.calls-before,evidence['output_tokens']+1);self.assertLess(evidence['measured_cdf_abs_error'],1e-5);self.assertFalse(evidence['qualification_claim']);self.assertFalse(evidence['cross_device_qualified'])
