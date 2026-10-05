"""Real tiny CPU cached inference controls; fake session is not MATH qualification."""
import copy,functools,time,unittest,base64
from types import SimpleNamespace
from unittest.mock import patch
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from nacl.signing import SigningKey
from subnet.owned_cached_evaluation import rollout,cohort,POLICY,paired_summary,digest
class OwnedCachedControls(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2);torch.manual_seed(7)
  c=Qwen2Config(vocab_size=127,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64);c._attn_implementation='eager';model=Qwen2ForCausalLM(c).eval()
  self.closed=0;owner=self
  class Session:
   def reset(self,index,seed):return dict(messages=[{'role':'user','content':'math prompt'}],tools=[],task_hash=f'{index:064x}')
   def step(self,action):return dict(done=True,reward=1.,classification='positive',observations=[])
   def close(self):owner.closed+=1
  self.session=Session;self.r=SimpleNamespace(model=model,tokenizer=SimpleNamespace(eos_token_id=None,decode=lambda ids,**kw:'answer'),harness=dict(version='text-tools-long-kv-v3',policy='autoregressive',max_output_tokens=8,temperature=.8,top_p=.9),spec=SimpleNamespace(config={'seed':0},id='affine_math',max_turns=1),sampling_context=None,prompt=lambda messages,tools:[3,4,5,6])
 def runrow(self):return rollout(self.r,33,19,create_session=lambda spec:self.session())
 def test_actual_cache_forward_shape_no_prefill_or_proof_calls(self):
  calls=[];forward=self.r.model.forward
  @functools.wraps(forward)
  def spy(ids,**kw):calls.append((ids.shape[1],kw.get('past_key_values')is not None));return forward(ids,**kw)
  self.r.model.forward=spy;row=self.runrow()
  self.assertEqual(calls,[(4,False)]+[(1,True)]*7);self.assertEqual(row['generation_forward_calls'],8);self.assertEqual(row['generation_input_tokens'],11);self.assertEqual(row['cached_decode_calls'],7)
  self.assertFalse(row['verified']);self.assertTrue(row['native_graded']);self.assertFalse(row['proof_verification_performed']);self.assertEqual((row['prefill_probability_replay_calls'],row['TOPLOC_build_calls'],row['TOPLOC_verify_calls']),(0,0,0));self.assertEqual(self.closed,1);self.assertFalse(torch.cuda.is_initialized())
 def test_seed_output_reproducible_and_cache_resets_each_task(self):
  one=self.runrow();two=self.runrow();self.assertEqual(one['turns'],two['turns']);self.assertEqual(two['generation_input_tokens'],11);self.assertEqual(self.closed,2)
 def test_legacy_harness_or_forced_miner_context_refused(self):
  self.r.harness['version']='text-tools-long-v2'
  with self.assertRaises(ValueError):self.runrow()
  self.r.harness['version']='text-tools-long-kv-v3';self.r.sampling_context={'contract':'miner'}
  with self.assertRaises(ValueError):self.runrow()
 def test_native_grader_error_closes_session_not_wrong_answer(self):
  def broken(action):raise RuntimeError('grader infrastructure error')
  original=self.session
  def create(spec):s=original();s.step=broken;return s
  with self.assertRaises(RuntimeError):rollout(self.r,33,19,create_session=create)
  self.assertEqual(self.closed,1)
 def test_context_budget_refused_before_inference(self):
  self.r.harness['max_output_tokens']=64
  with self.assertRaises(ValueError):self.runrow()
  self.assertEqual(self.closed,1)
 def test_pair_cohort_and_assurance_not_mixed_with_original_verified_harness(self):
  one=self.runrow();one.update(cohort_sha256='a'*64,checkpoint='b'*64);two=copy.deepcopy(one);two['checkpoint']='c'*64
  summary=paired_summary([one],[two]);self.assertEqual(summary['baseline_correct'],1);self.assertFalse(summary['proofs_verified'])
  for field,value in [('cohort_sha256','d'*64),('task_hash','f'*64),('seed',20),('verified',True)]:
   bad=copy.deepcopy(two);bad[field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):paired_summary([one],[bad])
 def test_cohort_hash_binds_harness_runtime_grader_source_and_order(self):
  definition=dict(env_id='affine_math',indices=[1,2],spec={'id':'affine_math','source_hash':'a'*64});suite=dict(env_id='affine_math',indices=[33,34],seeds=[19,20],harness=self.r.harness);manifest={'model_runtime_revision':'cpu-test','backend_profile':{'dtype':'float32'}};_,h=cohort(definition,suite,manifest,{'x.py':'b'*64})
  for changed in [dict(suite,seeds=[20,19]),dict(suite,harness=dict(self.r.harness,temperature=.7))]:self.assertNotEqual(cohort(definition,changed,manifest,{'x.py':'b'*64})[1],h)
  with self.assertRaises(ValueError):cohort(definition,dict(suite,indices=[1,34]),manifest,{})
class SignedOwnedJobControls(unittest.TestCase):
 def setUp(self):
  from subnet.backend_jobs import REVISION,NUMERICAL_POLICY,BACKEND_PROFILE,SOURCE_FILES,file_map
  self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex();files={'config.json':'1'*64,'model.safetensors':'2'*64}
  self.m={'epoch':'nonpayable-test','checkpoint':{'id':file_map(files),'files':files},'model_runtime_revision':REVISION,'numerical_policy':NUMERICAL_POLICY,'backend_profile':BACKEND_PROFILE,'K':1,'L':1,'audit_policy':{'mode':'full'}}
  self.job={'schema':1,'job_id':'cached-eval-new-cohort','role':'evaluate','created_at':10,'expires_at':100,'manifest':self.sign(self.m),'source_files':{n:'a'*64 for n in SOURCE_FILES},'runtime_versions':{'torch':'approved','transformers':'approved','toploc':'approved'},'heldout':[dict(env_id='affine_math',indices=[33],seeds=[19],harness={'version':'text-tools-long-kv-v3','policy':'autoregressive'})],'owned_evaluation_policy':POLICY}
 def sign(self,p):return dict(payload=copy.deepcopy(p),signer=self.authority,signature=base64.b64encode(self.key.sign(__import__('subnet.backend_jobs',fromlist=['canonical']).canonical(p)).signature).decode())
 def test_signed_opt_in_and_default_old_evaluator_policy(self):
  from subnet.backend_jobs import validate
  self.assertEqual(validate(self.sign(self.job),self.authority,now=50)[0]['owned_evaluation_policy'],POLICY)
  del self.job['owned_evaluation_policy'];self.job['heldout'][0]['harness']['version']='text-tools-long-v2';self.assertNotIn('owned_evaluation_policy',validate(self.sign(self.job),self.authority,now=50)[0])
 def test_wrong_role_harness_source_or_assurance_refused(self):
  from subnet.backend_jobs import validate
  for kind in ('role','harness','source','proof'):
   job=copy.deepcopy(self.job)
   if kind=='role':job['role']='train'
   if kind=='harness':job['heldout'][0]['harness']['version']='text-tools-long-v2'
   if kind=='source':del job['source_files']['subnet/owned_cached_evaluation.py']
   if kind=='proof':job['owned_evaluation_policy']['proof_reverification']=True
   with self.subTest(kind=kind),self.assertRaises((ValueError,KeyError)):validate(self.sign(job),self.authority,now=50)
if __name__=='__main__':unittest.main()
