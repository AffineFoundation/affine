import copy,json,sqlite3,unittest
from nacl.signing import SigningKey
from ops.verifier_workforce import authenticate_supplements,authorize_worker
from ops.live_reward_exporter import sign
from subnet.live_reward_bridge import sha
class WorkforceTests(unittest.TestCase):
 def setUp(self):
  self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex();self.old=['1'*64,'2'*64];self.new=['3'*64,'4'*64,'5'*64,'6'*64];self.cutover='a'*64
  self.manifest={'epoch':'nonpayable-live-reward-math-v1--123-9','live_reward_contract':{'version':'original'},'source_bundle':{'sha256':'b'*64},'backend_profile':{'sm':[9,0]},'numerical_policy':{'atol':1e-5},'checkpoint':{'id':'c'*64,'files':{'config.json':'d'*64}}}
  self.job={'source_files':{'subnet/backend_jobs.py':'e'*64},'runtime_versions':{'torch':'2.14.0','transformers':'5.14.1','toploc':'0.1.6'}}
  self.payload={'version':'operational-verifier-workforce-v1','cutover_document_sha256':self.cutover,'epoch':self.manifest['epoch'],'opening_manifest_sha256':sha(self.manifest),'live_reward_contract_sha256':sha(self.manifest['live_reward_contract']),'source_sha256':'b'*64,'source_files':self.job['source_files'],'runtime_versions':self.job['runtime_versions'],'backend_profile':self.manifest['backend_profile'],'numerical_policy':self.manifest['numerical_policy'],'checkpoint':self.manifest['checkpoint'],'existing_verifier_identities':self.old,'additional_verifier_identities':self.new,'role':'verify','effective_at':100,'previous_supplement_sha256':None}
  self.db=sqlite3.connect(':memory:');self.db.row_factory=sqlite3.Row;self.db.execute('create table events(sequence integer primary key,job text,at real,kind text,detail text)');self.claim(101);self.row={'id':'job','attempt':1,'report_request':json.dumps({'payload':{'at':110}})}
 def tearDown(self):self.db.close()
 def claim(self,at,worker=None,attempt=1):self.db.execute('insert into events(job,at,kind,detail)values(?,?,?,?)',('job',at,'claimed',json.dumps({'worker':worker or self.new[0],'attempt':attempt})))
 def supplements(self,p=None):return authenticate_supplements([sign(p or self.payload,self.key)],self.authority,self.cutover,self.old)
 def authorize(self,worker=None,manifest=None,job=None,supplements=None):return authorize_worker(worker or self.new[0],manifest or self.manifest,job or self.job,self.row,self.db,self.old,self.supplements()if supplements is None else supplements)
 def test_real_signature_union_keeps_old_and_exact_new_claim(self):
  result=self.authorize();self.assertEqual(result['worker_claimed_at'],101);self.assertEqual(result['workforce_authorized_at'],100)
  self.assertIsNone(self.authorize(worker=self.old[0]));self.assertIsNone(self.authorize(worker=self.old[1],supplements={}))
 def test_forged_authority_payload_or_unknown_fields_refuse(self):
  d=sign(self.payload,self.key);d['payload']['role']='train'
  with self.assertRaises(Exception):authenticate_supplements([d],self.authority,self.cutover,self.old)
  with self.assertRaises(Exception):authenticate_supplements([sign(self.payload,SigningKey.generate())],self.authority,self.cutover,self.old)
  for change in [{'role':'train'},{'unexpected':True},{'cutover_document_sha256':'f'*64},{'effective_at':True},{'effective_at':float('nan')},{'additional_verifier_identities':[]},{'additional_verifier_identities':[self.old[0],*self.new[1:]]},{'existing_verifier_identities':['7'*64,'8'*64]}]:
   with self.subTest(change=change),self.assertRaises(Exception):self.supplements(dict(self.payload,**change))
 def test_wrong_epoch_source_contract_cp_runtime_policy_refuse(self):
  for field,value in [('epoch','nonpayable-live-reward-math-v1-other'),('source_bundle',{'sha256':'f'*64}),('live_reward_contract',{'changed':True}),('checkpoint',{'id':'c'*64,'files':{'config.json':'f'*64}}),('numerical_policy',{'atol':1}),('backend_profile',{'sm':[8,6]})]:
   m=copy.deepcopy(self.manifest);m[field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):self.authorize(manifest=m)
  for field,value in [('source_files',{'subnet/backend_jobs.py':'f'*64}),('runtime_versions',{'torch':'other','transformers':'5.14.1','toploc':'0.1.6'})]:
   with self.assertRaises(ValueError):self.authorize(job=dict(self.job,**{field:value}))
 def test_unauthorized_unknown_worker_and_no_supplement_refuse(self):
  for worker,s in [(self.new[0],{}),('7'*64,self.supplements())]:
   with self.assertRaises(ValueError):self.authorize(worker=worker,supplements=s)
 def test_preauthorization_claim_no_claim_wrong_attempt_duplicates_refuse(self):
  for at,attempt in [(99,1),(101,2)]:
   self.db.execute('delete from events');self.claim(at,attempt=attempt)
   with self.assertRaises(ValueError):self.authorize()
  self.db.execute('delete from events')
  with self.assertRaises(ValueError):self.authorize()
  self.claim(101);self.claim(102)
  with self.assertRaises(ValueError):self.authorize()
 def test_prior_newworker_claim_and_report_time_refuse(self):
  self.claim(99,attempt=2)
  with self.assertRaises(ValueError):self.authorize()
  self.db.execute('delete from events');self.claim(111)
  with self.assertRaises(ValueError):self.authorize()
 def test_duplicate_epoch_supplement_refuses(self):
  doc=sign(self.payload,self.key)
  with self.assertRaises(ValueError):authenticate_supplements([doc,doc],self.authority,self.cutover,self.old)
class StagedWorkforceTests(unittest.TestCase):
 setUp=WorkforceTests.setUp
 tearDown=WorkforceTests.tearDown
 claim=WorkforceTests.claim
 authorize=WorkforceTests.authorize
 supplements=WorkforceTests.supplements
 def test_valid_addition_preserves_first_worker_earliest_claim_authority(self):
  first=dict(self.payload,additional_verifier_identities=self.new[:1]);doc=sign(first,self.key)
  second=dict(self.payload,additional_verifier_identities=self.new[:2],effective_at=105,previous_supplement_sha256=sha(doc));doc2=sign(second,self.key)
  supplements=authenticate_supplements([doc,doc2],self.authority,self.cutover,self.old)
  actual=self.authorize(supplements=supplements);self.assertEqual(actual['workforce_authorized_at'],100);self.assertEqual(actual['workforce_supplement_sha256'],sha(doc))
  self.claim(106,worker=self.new[1]);self.assertEqual(self.authorize(worker=self.new[1],supplements=supplements)['workforce_authorized_at'],105)
  self.claim(104,worker=self.new[1],attempt=2)
  with self.assertRaises(ValueError):self.authorize(worker=self.new[1],supplements=supplements)
 def test_branch_remove_reorder_context_time_regressions_refuse(self):
  first=dict(self.payload,additional_verifier_identities=self.new[:2]);doc=sign(first,self.key)
  good=dict(self.payload,additional_verifier_identities=self.new[:3],effective_at=105,previous_supplement_sha256=sha(doc))
  for change in [{'previous_supplement_sha256':'f'*64},{'additional_verifier_identities':self.new[1:3]},{'additional_verifier_identities':self.new[:2]},{'effective_at':99},{'opening_manifest_sha256':'f'*64},{'numerical_policy':{'atol':1}}]:
   with self.subTest(change=change),self.assertRaises(ValueError):authenticate_supplements([doc,sign(dict(good,**change),self.key)],self.authority,self.cutover,self.old)
  with self.assertRaises(ValueError):authenticate_supplements([sign(good,self.key),doc],self.authority,self.cutover,self.old)
  doc2=sign(good,self.key)
  with self.assertRaises(ValueError):authenticate_supplements([doc,doc2,sign(dict(good,additional_verifier_identities=self.new,effective_at=106),self.key)],self.authority,self.cutover,self.old)
if __name__=='__main__':unittest.main()
