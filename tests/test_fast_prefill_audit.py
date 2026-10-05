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
