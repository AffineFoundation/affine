"""Synthetic signed historical audit lineage; no real model/replay claim."""
import base64,copy,hashlib,io,json,unittest,zipfile
from nacl.signing import SigningKey
from subnet import verified_replay_pool as p

class ReplayTests(unittest.TestCase):
 def setUp(self):
  self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
  files={'config.json':'1'*64,'model.safetensors':'2'*64,'tokenizer.json':'3'*64,'tokenizer_config.json':'4'*64}
  self.historical={'epoch':'historical','checkpoint':{'id':p.digest(files),'files':files},'model_id':'approved-model','tokenizer_binding':{k:v for k,v in files.items() if k.startswith('tokenizer')},'source_bundle':{'key':'public/source.tar.gz','sha256':'5'*64,'size':100},'audit_policy':{'mode':'full'},'environments':[{'env_id':'env-'+f,'spec':{'id':'env-'+f,'version':f+'-v1','adapter':f,'config':{},'num_samples':32},'harness':{'version':'text-tools-v1','policy':'candidates','candidates':['yes','no'],'max_output_tokens':16,'temperature':1.,'top_p':1.},'indices':list(range(32))} for f in ['familyA','familyB','familyC']]}
  self.current=copy.deepcopy(self.historical);self.current['epoch']='current';self.current['checkpoint']['files']['model.safetensors']='6'*64;self.current['checkpoint']['id']=p.digest(self.current['checkpoint']['files']);self.current['replay_policy']={'max_pairs':3,'max_reuse':2,'max_zip_bytes':100000000,'reference_policy':p.REFERENCE_POLICY};self.current['heldout_indices']={r['env_id']:[31] for r in self.current['environments']};[r.update(indices=list(range(31))) for r in self.current['environments']];self.current['model_geometry']={'vocab_size':100,'max_context':8192,'max_output_tokens':128}
 def sign(self,value,key=None):
  key=key or self.key
  return {'payload':copy.deepcopy(value),'signer':key.verify_key.encode().hex(),'signature':base64.b64encode(key.sign(p.canonical(value)).signature).decode()}
 def fixture(self,family='familyA',index=0):
  env='env-'+family;row=next(r for r in self.historical['environments'] if r['env_id']==env);task=hashlib.sha256(f'{env}:{index}'.encode()).hexdigest()
  rolls=[{'classification':label,'env_id':env,'environment_version':row['spec']['version'],'index':index,'task_hash':task,'turns':[{'prompt':[1,2],'output':[20+number],'proofs':['synthetic-proof']}]} for number,label in enumerate(['positive','negative'])]
  batch={'schema':2,'epoch':'historical','checkpoint':self.historical['checkpoint']['id'],'env_id':env,'environment_version':row['spec']['version'],'index':index,'sample_index':index,'rollouts':rolls}
  buf=io.BytesIO()
  import numpy as np
  array=io.BytesIO();np.save(array,np.zeros((1,100),dtype=np.float32),allow_pickle=False)
  with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
   z.writestr('0-0-0.npy',array.getvalue());z.writestr('0-1-0.npy',array.getvalue());z.writestr('manifest.json',p.canonical([{'batch':batch,'arrays':[['0-0-0.npy'],['0-1-0.npy']]}]))
  data=buf.getvalue();audit={'epoch':'historical','training_eligibility':'fully-audited-only','submission_sha256':hashlib.sha256(data).hexdigest(),'outcomes':[{'batch':0,'valid':True,'fully_audited':True,'env_id':env,'index':index}],'accepted':[batch]};hm=self.sign(self.historical);ha=self.sign(audit)
  target={'environment_id':env,'environment_index':index,'task_hash':task,'positive_rollout_sha256':p.digest(rolls[0]),'negative_rollout_sha256':p.digest(rolls[1])}
  descriptor={'version':p.VERSION,'reference_policy':p.REFERENCE_POLICY,'auxiliary_model_roles':False,'historical_authority':self.authority,'historical_manifest_sha256':p.digest(hm),'historical_audit_sha256':p.digest(ha),'compatibility':p.compatibility(self.historical),'current_model_geometry':self.current['model_geometry'],'historical_checkpoint':self.historical['checkpoint'],'source_bundle':self.historical['source_bundle'],'frozen_zip_sha256':hashlib.sha256(data).hexdigest(),'frozen_zip_size':len(data),'batch_number':0,'family':env,'adapter':row['spec']['adapter'],'environment_id':env,'environment_index':index,'environment':row['spec'],'harness':row['harness'],'batch_sha256':p.digest(batch),**target,'target_sha256':p.digest(target)}
  return self.sign(descriptor),hm,ha,data
 def validate(self,fixture=None):
  d,hm,ha,data=fixture or self.fixture();return p.validate_entry(d,self.authority,hm,ha,data,self.sign(self.current))
 def test_full_signed_history_accepts_curated_reference_recompute_only(self):
  entry=self.validate();self.assertEqual(entry['current_checkpoint']['id'],self.current['checkpoint']['id']);self.assertNotEqual(entry['current_checkpoint']['id'],entry['historical_checkpoint']['id']);self.assertFalse(entry['fresh_numerical_verification_performed']);self.assertFalse(entry['historical_probabilities_are_current_reference'])
 def test_forged_audit_and_modified_report_rejected(self):
  d,hm,ha,data=self.fixture();ha['payload']['outcomes'][0]['valid']=False
  with self.assertRaises(Exception):self.validate((d,hm,ha,data))
  d,hm,ha,data=self.fixture();other=SigningKey.generate();ha=self.sign(ha['payload'],other)
  with self.assertRaisesRegex(ValueError,'authority'):self.validate((d,hm,ha,data))
 def test_changed_pair_hash_or_frozen_bytes_rejected(self):
  d,hm,ha,data=self.fixture();d['payload']['positive_rollout_sha256']='a'*64
  with self.assertRaisesRegex(ValueError,'rollout hash'):self.validate((self.sign(d['payload']),hm,ha,data))
  d,hm,ha,data=self.fixture()
  with self.assertRaisesRegex(ValueError,'SHA/size'):self.validate((d,hm,ha,data+b'changed'))
 def test_heldout_overlap_and_unsigned_task_index_rejected(self):
  with self.assertRaisesRegex(ValueError,'heldout exclusion'):self.validate(self.fixture(index=31))
  self.current['environments'][0]['indices']=[0]
  with self.assertRaisesRegex(ValueError,'heldout exclusion'):self.validate(self.fixture(index=1))
 def test_wrong_tokenizer_config_or_architecture_rejected(self):
  for field in ['tokenizer.json','config.json','model_id']:
   self.setUp()
   if field=='model_id':self.current[field]='different-architecture'
   else:
    self.current['checkpoint']['files'][field]='a'*64;self.current['checkpoint']['id']=p.digest(self.current['checkpoint']['files'])
    if field.startswith('tokenizer'):self.current['tokenizer_binding'][field]='a'*64
   with self.subTest(field=field),self.assertRaisesRegex(ValueError,'incompatibility'):self.validate()
 def test_sampled_audit_and_native_auxiliary_geometry_fail_closed(self):
  self.historical['audit_policy']['mode']='sampled'
  with self.assertRaisesRegex(ValueError,'sampled'):self.validate()
  self.historical['audit_policy']['mode']='full';self.historical['environments'][0]['spec']['adapter']='native_tau2';self.current['environments'][0]['spec']['adapter']='native_tau2'
  with self.assertRaisesRegex(ValueError,'native auxiliary'):self.validate()
 def test_dominated_family_inventory_selected_round_robin(self):
  entries=[self.sign(self.validate(self.fixture('familyA',i))) for i in range(6)]+[self.sign(self.validate(self.fixture('familyB',10))),self.sign(self.validate(self.fixture('familyC',20)))]
  pool=p.build_pool(entries,self.sign(self.current),self.authority);result=p.select_pool(self.sign(pool),self.authority,{})
  self.assertEqual([e['family'] for e in result['selected']],['env-familyA','env-familyB','env-familyC']);self.assertEqual(len(result['selected']),3)
  self.assertEqual(result['recompute_reference_checkpoint']['id'],self.current['checkpoint']['id'])
 def test_reuse_cap_uniqueness_and_signed_pool_digest(self):
  entry=self.validate();signed=self.sign(entry)
  with self.assertRaisesRegex(ValueError,'unique'):p.build_pool([signed,signed],self.sign(self.current),self.authority)
  pool=p.build_pool([signed],self.sign(self.current),self.authority);result=p.select_pool(self.sign(pool),self.authority,{entry['target_sha256']:2});self.assertEqual(result['selected'],[])
  pool['entries'][0]['environment_index']=5
  with self.assertRaisesRegex(ValueError,'pool digest'):p.select_pool(self.sign(pool),self.authority,{})
 def test_plain_mutable_views_are_not_admission_authority(self):
  entry=self.validate()
  with self.assertRaises(ValueError):p.build_pool([entry],self.sign(self.current),self.authority)
  pool=p.build_pool([self.sign(entry)],self.sign(self.current),self.authority)
  with self.assertRaises(ValueError):p.select_pool(pool,self.authority,{})
 def test_token_geometry_and_budget_types_rejected(self):
  self.current['replay_policy']['max_pairs']=True
  with self.assertRaisesRegex(ValueError,'bounded'):self.validate()
  self.current['replay_policy']['max_pairs']=3;self.current['model_geometry']['vocab_size']=20
  with self.assertRaisesRegex(ValueError,'token geometry'):self.validate()

 def test_twelve_actual_style_prime_adapter_environments_each_get_first_turn(self):
  families=[f'family{i:02}' for i in range(12)]
  self.historical['environments']=[{'env_id':'env-'+f,'spec':{'id':'env-'+f,'version':f+'-v1','adapter':'prime_v1','config':{},'num_samples':32},'harness':{'version':'text-tools-v1','policy':'candidates','candidates':['yes','no'],'max_output_tokens':16,'temperature':1.,'top_p':1.},'indices':list(range(32))} for f in families]
  self.current['environments']=copy.deepcopy(self.historical['environments']);[r.update(indices=list(range(31))) for r in self.current['environments']];self.current['heldout_indices']={r['env_id']:[31] for r in self.current['environments']};self.current['replay_policy']['max_pairs']=12
  entries=[self.sign(self.validate(self.fixture(families[0],i))) for i in range(8)]+[self.sign(self.validate(self.fixture(f,10))) for f in families[1:]]
  pool=p.build_pool(entries,self.sign(self.current),self.authority);selected=p.select_pool(self.sign(pool),self.authority,{})['selected']
  self.assertEqual([e['environment_id'] for e in selected],['env-'+f for f in families]);self.assertEqual({e['adapter'] for e in selected},{'prime_v1'})

 def test_missing_or_extra_heldout_registry_keys_rejected(self):
  for mode in ['empty','missing-other','missing-target','extra']:
   self.setUp()
   if mode=='empty':self.current['heldout_indices']={}
   elif mode=='missing-other':self.current['heldout_indices'].pop('env-familyB')
   elif mode=='missing-target':self.current['heldout_indices'].pop('env-familyA')
   else:self.current['heldout_indices']['unregistered-environment']=[]
   with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,'complete explicit'):self.validate()

 def test_heldout_types_duplicates_bounds_and_other_family_overlap_rejected(self):
  for values in [[True],[31,31],[-1],[32],[0]]:
   self.setUp();self.current['heldout_indices']['env-familyB']=values
   with self.subTest(values=values),self.assertRaises(ValueError):self.validate()

 def test_explicit_empty_heldout_is_distinct_from_missing_registry(self):
  self.current['heldout_indices']['env-familyA']=[]
  self.assertEqual(self.validate()['environment_id'],'env-familyA')

 def test_signed_pool_cannot_bypass_missing_heldout_registry(self):
  entry=self.sign(self.validate());self.current['heldout_indices'].pop('env-familyC')
  with self.assertRaisesRegex(ValueError,'complete explicit'):
   p.build_pool([entry],self.sign(self.current),self.authority)
