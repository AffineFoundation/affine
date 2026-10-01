"""Synthetic runtime conformance, never genuine inference or native rewards."""
import copy,hashlib,unittest
import numpy as np
import test_native_tau2_common_contract as fixtures
from subnet import native_tau2_common_endpoint as e
from subnet import native_tau2_common_contract as c

class Tokenizer:
 def __init__(self,text):self.text=text;self.messages=None
 def apply_chat_template(self,messages,**kwargs):self.messages=copy.deepcopy(messages);return [1,2]
 def decode(self,output,**kwargs):return self.text
 def encode(self,text,**kwargs):return [8]
class Runtime:
 def __init__(self,descriptor,text='ordinary output'):self.descriptor=copy.deepcopy(descriptor);self.tokenizer=Tokenizer(text);self.calls=[]
 def approved_descriptor(self):return copy.deepcopy(self.descriptor)
 def sample(self,prompt,seed,temperature,top_p,max_tokens):self.calls.append((copy.deepcopy(prompt),seed));return [20 if self.descriptor['kind']=='agent' else 10]
 def compute(self,prompt,output):return 'synthetic-activations',np.zeros((len(output),self.descriptor['vocab_size']),dtype=np.float32)
 def build_proofs(self,activations,**kwargs):return ['synthetic-prefix','synthetic-output']
 def verify_proofs(self,activations,proofs,**kwargs):
  from types import SimpleNamespace
  return [SimpleNamespace(exp_mismatches=0,mant_err_mean=0,mant_err_median=0) for _ in proofs]
class EndpointTests(unittest.TestCase):
 def setUp(self):
  self.fixture=fixtures.ContractTests();self.fixture.setUp();f=self.fixture
  self.runtimes={'agent':Runtime(f.agent),'user':Runtime(f.user)};self.sources=[]
  self.endpoint=e.CommonRoleEndpoint(f.sign(f.manifest),f.authority,f.user,self.runtimes,f.key,0,source_validator=lambda d:self.sources.append(c.digest(d)),clock=lambda:100.5)
 def request(self,name):return {'model':self.fixture.manifest['roles'][name]['request_model'],'messages':[{'role':'user','content':'Complete public context'}],'tools':[{'type':'function','function':{'name':'lookup','description':'ALL_SCHEMA','parameters':{'type':'object','properties':{'key':{'type':'string'}}}}}]}
 def test_two_role_models_seeds_masks_and_contract_receipt_fields(self):
  user,ur,ua=self.endpoint.response(self.request('user'));agent,ar,aa=self.endpoint.response(self.request('agent'));_,ur2,_=self.endpoint.response(self.request('user'))
  self.assertEqual(self.runtimes['user'].calls,[([1,2],57),([1,2],58)]);self.assertEqual(self.runtimes['agent'].calls,[([1,2],57)])
  self.assertEqual(ur['payload']['loss_mask'],[False]);self.assertEqual(ar['payload']['loss_mask'],[True]);self.assertEqual(ur2['payload']['ordinal'],2);self.assertEqual(ur2['payload']['role_ordinal'],1)
  self.assertEqual(ur['payload']['checkpoint'],self.fixture.user['checkpoint']);self.assertEqual(ar['payload']['checkpoint'],self.fixture.agent['checkpoint'])
  self.assertEqual(hashlib.sha256(ua).hexdigest(),ur['payload']['probabilities_sha256']);c.authenticate(ur,self.fixture.authority);self.assertTrue(e.validate_response_binding(ar['payload']))
 def test_original_renderer_preserves_all_messages_and_full_tool_schema(self):
  request=self.request('agent');request['messages'].append({'role':'tool','tool_call_id':'actual-id','content':'Exact public tool observation'})
  self.endpoint.response(request);messages=self.runtimes['agent'].tokenizer.messages
  self.assertIn('ALL_SCHEMA',messages[0]['content']);self.assertIn('properties',messages[0]['content']);self.assertIn('actual-id',messages[-1]['content']);self.assertIn('Exact public tool observation',messages[-1]['content'])
 def test_unknown_role_and_runtime_drift_rejected_before_model_call(self):
  request=self.request('agent');request['model']='hidden-model'
  with self.assertRaisesRegex(ValueError,'request model role'):self.endpoint.response(request)
  self.runtimes['agent'].descriptor['checkpoint']=self.fixture.user['checkpoint']
  with self.assertRaisesRegex(ValueError,'runtime role drift'):self.endpoint.response(self.request('agent'))
  self.assertEqual(self.runtimes['agent'].calls,[])
 def test_source_failure_precedes_inference(self):
  self.endpoint.source_validator=lambda _:(_ for _ in ()).throw(ValueError('source mismatch'))
  with self.assertRaisesRegex(ValueError,'source mismatch'):self.endpoint.response(self.request('user'))
  self.assertEqual(self.runtimes['user'].calls,[])
 def test_context_overflow_rejects_without_truncation(self):
  self.endpoint.renderer=lambda *_:[1]*8192
  with self.assertRaisesRegex(ValueError,'context exceeds'):self.endpoint.response(self.request('agent'))
  self.assertEqual(self.runtimes['agent'].calls,[]);self.assertEqual(self.endpoint.records,[])
 def test_precise_tool_response_and_last_response_mutations_rejected(self):
  self.runtimes['agent'].tokenizer.text='{"tool_call":{"name":"lookup","arguments":{"key":"public"}}}'
  response,receipt,_=self.endpoint.response(self.request('agent'));r=receipt['payload'];self.assertEqual(response['choices'][0]['finish_reason'],'tool_calls');self.assertTrue(e.validate_response_binding(r))
  for field in ('content','arguments','usage'):
   bad=copy.deepcopy(r)
   if field=='content':bad['response']['choices'][0]['message']['content']='forged user observation'
   elif field=='arguments':bad['response']['choices'][0]['message']['tool_calls'][0]['function']['arguments']='{"key":"forged"}'
   else:bad['response']['usage']['completion_tokens']=3
   bad['response_sha256']=c.digest(bad['response'])
   with self.subTest(field=field),self.assertRaisesRegex(ValueError,'derived native response'):e.validate_response_binding(bad)
 def test_auxiliary_mask_and_prompt_hash_tamper_rejected(self):
  _,receipt,_=self.endpoint.response(self.request('user'));r=receipt['payload'];bad=copy.deepcopy(r);bad['loss_mask']=[True]
  with self.assertRaisesRegex(ValueError,'loss mask'):e.validate_response_binding(bad)
  bad=copy.deepcopy(r);bad['prompt']=[3]
  with self.assertRaises(ValueError):e.validate_response_binding(bad)
 def test_unpinned_candidate_policy_rejected_at_construction(self):
  class Policy:
   def approved_descriptor(self):return {'scope':'public-request-derived-curated-output'}
  f=self.fixture
  with self.assertRaisesRegex(ValueError,'pinned public'):e.CommonRoleEndpoint(f.sign(f.manifest),f.authority,f.user,self.runtimes,f.key,0,source_validator=lambda _:None,candidate_policy=Policy())
 def test_receipts_admit_only_after_separate_signed_complete_audit(self):
  records=[]
  for name in ['user','agent']:records.append(self.endpoint.response(self.request(name))[1])
  f=self.fixture;_,_,report=f.sample();report['role_checks']=[{'signed_receipt_sha256':c.digest(r),'role':r['payload']['role'],'model_computation_verified':True,'context_verified':True,'derived_response_verified':True} for r in records]
  audit=f.audit(records,report);result=c.admit_sample(f.sign(f.manifest),records,audit,report,f.authority,f.user)
  self.assertEqual([v['loss_mask'] for v in result['training_view']],[[False],[True]])
  report['role_checks'][0]['derived_response_verified']=False
  with self.assertRaises(ValueError):c.admit_sample(f.sign(f.manifest),records,f.audit(records,report),report,f.authority,f.user)

 def test_output_arrays_and_signed_receipts_are_private_exact_bytes(self):
  import tempfile,pathlib,stat
  f=self.fixture
  with tempfile.TemporaryDirectory() as tmp:
   out=pathlib.Path(tmp)/'artifacts';endpoint=e.CommonRoleEndpoint(f.sign(f.manifest),f.authority,f.user,self.runtimes,f.key,0,source_validator=lambda _:None,artifact_dir=out)
   _,receipt,array=endpoint.response(self.request('user'))
   self.assertEqual((out/'role-0.npy').read_bytes(),array);self.assertEqual((out/'role-0.json').read_bytes(),c.canonical(receipt))
   self.assertEqual(stat.S_IMODE((out/'role-0.npy').stat().st_mode),0o600);self.assertEqual(stat.S_IMODE(out.stat().st_mode),0o700)
 def test_pinned_agent_public_policy_never_controls_auxiliary_user(self):
  f=self.fixture;descriptor={'scope':'public-request-derived-curated-output','source_files':{'subnet/public-policy.py':'d'*64}}
  f.manifest['roles']['agent']['candidate_policy']=descriptor;f.manifest['roles']['agent']['source_files'].update(descriptor['source_files']);self.runtimes['agent'].descriptor=copy.deepcopy(f.manifest['roles']['agent'])
  class Policy:
   calls=[]
   def approved_descriptor(self):return descriptor
   def select(self,request,seed):self.calls.append((request,seed));return 'public output'
  policy=Policy();endpoint=e.CommonRoleEndpoint(f.sign(f.manifest),f.authority,f.user,self.runtimes,f.key,0,source_validator=lambda _:None,candidate_policy=policy)
  _,agent,_=endpoint.response(self.request('agent'));_,user,_=endpoint.response(self.request('user'))
  self.assertEqual(len(policy.calls),1);self.assertEqual(policy.calls[0][1],57);self.assertEqual(agent['payload']['output'],[8]);self.assertEqual(user['payload']['output'],[10]);self.assertEqual(self.runtimes['agent'].calls,[])

 def verify(self,receipt,array,runtime=None):
  f=self.fixture
  return e.verify_receipt(f.sign(f.manifest),receipt,array,f.authority,f.user,runtime or self.runtimes[receipt['payload']['role']],0,source_validator=lambda _:None,framing_validator=lambda proof,count:self.assertEqual(len(proof),count))
 def test_independent_role_recomputation_and_wrong_model_rejected(self):
  _,receipt,array=self.endpoint.response(self.request('agent'));fresh=Runtime(self.fixture.agent)
  result=self.verify(receipt,array,fresh);self.assertTrue(result['model_computation_verified']);self.assertFalse(result['native_replay_performed_here'])
  fresh.descriptor=self.fixture.user
  with self.assertRaisesRegex(ValueError,'runtime role descriptor'):self.verify(receipt,array,fresh)
 def test_fresh_verifier_rejects_resigned_context_response_and_seed(self):
  _,receipt,array=self.endpoint.response(self.request('user'))
  for field in ('prompt','response','seed'):
   record=copy.deepcopy(receipt['payload'])
   if field=='prompt':record['prompt']=[3,4];record['prompt_sha256']=c.digest(record['prompt'])
   elif field=='response':record['response']['choices'][0]['message']['content']='forged';record['response_sha256']=c.digest(record['response'])
   else:record['seed']+=1
   with self.subTest(field=field),self.assertRaises(ValueError):self.verify(self.fixture.sign(record),array)
 def test_fresh_verifier_rejects_self_consistent_probability_forgery(self):
  import io
  _,receipt,array=self.endpoint.response(self.request('agent'));record=copy.deepcopy(receipt['payload']);buf=io.BytesIO();np.save(buf,np.ones((1,100),dtype=np.float32),allow_pickle=False);array=buf.getvalue();record['probabilities_sha256']=hashlib.sha256(array).hexdigest()
  with self.assertRaisesRegex(ValueError,'probability recomputation'):self.verify(self.fixture.sign(record),array)
 def test_fresh_verifier_rejects_nonzero_toploc_errors(self):
  from types import SimpleNamespace
  _,receipt,array=self.endpoint.response(self.request('agent'));fresh=Runtime(self.fixture.agent)
  fresh.verify_proofs=lambda *args,**kwargs:[SimpleNamespace(exp_mismatches=1,mant_err_mean=0,mant_err_median=0)]*2
  with self.assertRaisesRegex(ValueError,'TOPLOC'):self.verify(receipt,array,fresh)

 def test_fixed_user_public_policy_is_immutable_and_masked(self):
  f=self.fixture;descriptor={'scope':'public-request-derived-curated-output','source_files':{'subnet/fixed-user-public.py':'e'*64}}
  f.user['candidate_policy']=descriptor;f.user['source_files'].update(descriptor['source_files']);self.runtimes['user']=Runtime(f.user)
  class Policy:
   def approved_descriptor(self):return descriptor
   def select(self,request,seed):return 'fixed public user output'
  policy=Policy();endpoint=e.CommonRoleEndpoint(f.sign(f.manifest),f.authority,f.user,self.runtimes,f.key,0,source_validator=lambda _:None,user_candidate_policy=policy)
  _,receipt,array=endpoint.response(self.request('user'));self.assertEqual(receipt['payload']['loss_mask'],[False]);self.assertEqual(receipt['payload']['generation_scope'],'fixed-user-public-request-derived-curated-output');self.assertEqual(self.runtimes['user'].calls,[])
  verified=e.verify_receipt(f.sign(f.manifest),receipt,array,f.authority,f.user,Runtime(f.user),0,source_validator=lambda _:None,framing_validator=lambda proof,count:self.assertEqual(len(proof),count),candidate_policy=policy)
  self.assertTrue(verified['model_computation_verified'])
  fixed=copy.deepcopy(f.user);manifest=copy.deepcopy(f.manifest);manifest['roles']['user']['candidate_policy']['source_files']['subnet/fixed-user-public.py']='f'*64
  with self.assertRaisesRegex(ValueError,'fixed auxiliary'):c.validate_epoch(f.sign(manifest),f.authority,fixed)
 def test_declared_policy_dependency_and_curated_output_cannot_be_bypassed(self):
  f=self.fixture;descriptor={'scope':'public-request-derived-curated-output','source_files':{'subnet/fixed-user-public.py':'e'*64}}
  f.user['candidate_policy']=descriptor;f.user['source_files'].update(descriptor['source_files']);self.runtimes['user']=Runtime(f.user)
  with self.assertRaisesRegex(ValueError,'dependency missing'):e.CommonRoleEndpoint(f.sign(f.manifest),f.authority,f.user,self.runtimes,f.key,0,source_validator=lambda _:None)
  class Policy:
   def approved_descriptor(self):return descriptor
   def select(self,request,seed):return 'fixed user output'
  policy=Policy();endpoint=e.CommonRoleEndpoint(f.sign(f.manifest),f.authority,f.user,self.runtimes,f.key,0,source_validator=lambda _:None,user_candidate_policy=policy)
  _,receipt,array=endpoint.response(self.request('user'));fresh=Runtime(f.user);fresh.tokenizer.encode=lambda *args,**kwargs:[9]
  with self.assertRaisesRegex(ValueError,'policy output mismatch'):e.verify_receipt(f.sign(f.manifest),receipt,array,f.authority,f.user,fresh,0,source_validator=lambda _:None,framing_validator=lambda *_:None,candidate_policy=policy)
