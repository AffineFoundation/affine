"""Synthetic signed conformance controls; no native/model execution claim."""
import base64,copy,unittest
from nacl.signing import SigningKey
from subnet import native_tau2_common_contract as c

class ContractTests(unittest.TestCase):
 def setUp(self):
  self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
  def role(kind,label):
   files={'config.json':'1'*64,'model.safetensors':label*64}
   return {'kind':kind,'training_eligible':kind=='agent','checkpoint':{'id':c.digest(files),'files':files},'source_files':{'subnet/model.py':'3'*64},'harness_source_sha256':'4'*64,'interpreter_sha256':'5'*64,'runtime_profile':{'dtype':'float32','threads':4},'runtime_versions':{'torch':'qualified-version'},'renderer':'complete-tools-v1','request_model':kind,'max_context':8192,'max_output_tokens':16,'vocab_size':100,'seed_policy':c.SEED_POLICY,'seed_start':50,'numerical_policy':{'logprobs_atol':1e-5,'logprobs_rtol':0,'TOPLOC_errors':0}}
  self.user=role('auxiliary','a');self.agent=role('agent','b')
  self.manifest={'version':c.VERSION,'objective':c.OBJECTIVE,'epoch':'epoch-1','checkpoint':self.agent['checkpoint'],'roles':{'agent':self.agent,'user':self.user},'environment':{'id':'tau2','version':'native-v1','taskset_sha256':'6'*64,'data_inventory_sha256':'7'*64,'source_files':{'subnet/native.py':'8'*64}},'tasks':[{'index':0,'task_hash':'9'*64,'seed':7}],'sampler_provenance':'curated-target-model-computation-only','payable':False,'chain_transactions':False}
 def sign(self,payload,key=None):
  key=key or self.key
  return {'payload':copy.deepcopy(payload),'signer':key.verify_key.encode().hex(),'signature':base64.b64encode(key.sign(c.canonical(payload)).signature).decode()}
 def sample(self,reward=1,output=20):
  records=[]
  for ordinal,name in enumerate(['user','agent']):
   role=self.manifest['roles'][name];prompt=[1,2,3];tokens=[10] if name=='user' else [output];request={'model':role['request_model'],'messages':[{'role':'user','content':'Full public context'}],'tools':[{'name':'complete-schema'}]};response={'content':str(tokens)}
   record={'manifest_sha256':c.digest(self.manifest),'epoch':'epoch-1','environment_id':'tau2','task_hash':'9'*64,'environment_index':0,'role_descriptor_sha256':c.digest(role),'ordinal':ordinal,'role_ordinal':0,'seed':57,'role':name,'checkpoint':role['checkpoint'],'runtime_profile':role['runtime_profile'],'harness_source_sha256':role['harness_source_sha256'],'renderer':role['renderer'],'source_files':role['source_files'],'request':request,'response':response,'request_sha256':c.digest(request),'response_sha256':c.digest(response),'prompt':prompt,'output':tokens,'prompt_sha256':c.digest(prompt),'output_sha256':c.digest(tokens),'probabilities_sha256':'c'*64,'proofs':['opaque-proof-conformance-only']}
   records.append(self.sign(record))
  report={'epoch':'epoch-1','environment_id':'tau2','environment_version':'native-v1','task_hash':'9'*64,'environment_index':0,'reward':reward,'role_checks':[{'signed_receipt_sha256':c.digest(r),'role':r['payload']['role'],'model_computation_verified':True,'context_verified':True,'derived_response_verified':True} for r in records]}
  for flag in ['full_native_trajectory_verified','all_model_roles_verified','derived_responses_verified','source_closure_verified']:report[flag]=True
  return records,self.audit(records,report),report
 def audit(self,records,report):
  value={'version':c.AUDIT_VERSION,'manifest_sha256':c.digest(self.manifest),'signed_receipts_sha256':c.digest(records),'verification_report_sha256':c.digest(report),'epoch':'epoch-1','environment_id':'tau2','environment_version':'native-v1','task_hash':'9'*64,'environment_index':0,'reward':report['reward'],'originally_sampled':False,'payable':False,'sampler_provenance':self.manifest['sampler_provenance']}
  for flag in ['full_native_trajectory_verified','all_model_roles_verified','derived_responses_verified','source_closure_verified']:value[flag]=True
  return self.sign(value)
 def admit(self,records,audit,report):return c.admit_sample(self.sign(self.manifest),records,audit,report,self.authority,self.user)
 def resign_record(self,records,report,index):
  records[index]=self.sign(records[index]['payload']);report['role_checks'][index]['signed_receipt_sha256']=c.digest(records[index]);return self.audit(records,report)
 def test_admission_masks_auxiliary_false_and_agent_true(self):
  records,audit,report=self.sample();sample=self.admit(records,audit,report)
  self.assertEqual([v['loss_mask'] for v in sample['training_view']],[[False],[True]])
  self.assertFalse(sample['originally_sampled']);self.assertFalse(sample['production_admitted'])
 def test_fixed_auxiliary_does_not_follow_current_agent_checkpoint(self):
  original=copy.deepcopy(self.user);files={'config.json':'1'*64,'model.safetensors':'d'*64}
  self.manifest['roles']['agent']['checkpoint']={'id':c.digest(files),'files':files};self.manifest['checkpoint']=self.manifest['roles']['agent']['checkpoint']
  c.validate_epoch(self.sign(self.manifest),self.authority,original)
  self.assertEqual(self.manifest['roles']['user'],original)
 def test_auxiliary_checkpoint_and_profile_drift_rejected(self):
  fixed=copy.deepcopy(self.user)
  for field in ['checkpoint','runtime_profile','harness_source_sha256','seed_start']:
   manifest=copy.deepcopy(self.manifest)
   if field=='checkpoint':
    files={'config.json':'1'*64,'model.safetensors':'d'*64};manifest['roles']['user'][field]={'id':c.digest(files),'files':files}
   elif field=='runtime_profile':manifest['roles']['user'][field]={'dtype':'float32','threads':8}
   elif field=='seed_start':manifest['roles']['user'][field]=51
   else:manifest['roles']['user'][field]='d'*64
   with self.subTest(field=field),self.assertRaisesRegex(ValueError,'fixed auxiliary'):c.validate_epoch(self.sign(manifest),self.authority,fixed)
 def test_malicious_auxiliary_loss_mask_rejected_even_with_bound_audit(self):
  records,audit,report=self.sample();records[0]['payload']['loss_mask']=[True];audit=self.resign_record(records,report,0)
  with self.assertRaisesRegex(ValueError,'loss mask'):self.admit(records,audit,report)
 def test_signer_substitution_rejected(self):
  records,audit,report=self.sample();other=SigningKey.generate();records[0]=self.sign(records[0]['payload'],other);audit=self.audit(records,report)
  with self.assertRaisesRegex(ValueError,'authority'):self.admit(records,audit,report)
 def test_tampered_report_and_missing_role_verification_rejected(self):
  records,audit,report=self.sample();report['reward']=0
  with self.assertRaisesRegex(ValueError,'lineage'):self.admit(records,audit,report)
  records,audit,report=self.sample();report['role_checks'][0]['model_computation_verified']=False;audit=self.audit(records,report)
  with self.assertRaisesRegex(ValueError,'proof/context audit'):self.admit(records,audit,report)
 def test_source_and_context_mismatch_rejected_even_with_resigned_receipt(self):
  for field in ['source_files','prompt_sha256','seed','checkpoint']:
   records,audit,report=self.sample();r=records[1]['payload']
   if field=='source_files':r[field]={'subnet/model.py':'e'*64}
   elif field=='seed':r[field]+=1
   elif field=='checkpoint':r[field]=self.user['checkpoint']
   else:r[field]='e'*64
   audit=self.resign_record(records,report,1)
   with self.subTest(field=field),self.assertRaises(ValueError):self.admit(records,audit,report)
 def test_same_prompt_positive_negative_agent_only_preference(self):
  pos=self.admit(*self.sample(1,20));neg=self.admit(*self.sample(0,21));pair=c.preference_pair(pos,neg)
  self.assertEqual(pair['chosen'],[20]);self.assertEqual(pair['rejected'],[21]);self.assertFalse(pair['auxiliary_tokens_in_loss'])
  neg['training_view'][1]['prompt']=[4]
  with self.assertRaisesRegex(ValueError,'same-prompt'):c.preference_pair(pos,neg)
 def test_serialized_training_view_cannot_smuggle_auxiliary_mask(self):
  pos=self.admit(*self.sample(1,20));neg=self.admit(*self.sample(0,21));neg['training_view'][0]['loss_mask']=[True]
  with self.assertRaisesRegex(ValueError,'loss mask'):c.preference_pair(pos,neg)
 def test_type_and_context_budget_are_strict(self):
  manifest=copy.deepcopy(self.manifest);manifest['roles']['user']['training_eligible']=0
  with self.assertRaisesRegex(ValueError,'eligibility'):c.validate_epoch(self.sign(manifest),self.authority,self.user)
  records,audit,report=self.sample();r=records[1]['payload'];r['prompt']=[1]*8192;r['prompt_sha256']=c.digest(r['prompt']);audit=self.resign_record(records,report,1)
  with self.assertRaisesRegex(ValueError,'context/token budget'):self.admit(records,audit,report)
