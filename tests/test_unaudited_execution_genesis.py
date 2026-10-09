import copy,json,unittest
from test_unaudited_execution_release import ReleaseTests
from subnet import unaudited_training_execution as a
from subnet import learning_rate_transition as lr
from subnet import committed_training_inputs as learner
from subnet.training_receipts import sha

class GenesisReleaseTests(ReleaseTests):
 def setUp(self):
  super().setUp()
  binding=self.f.manifest['payload']['trainer_state_binding']
  self.genesis=lr.genesis_document(binding['parameters_sha256'],binding['input_checkpoint'],'7'*64,self.release['payload']['effective_learning_rate'])
  r=copy.deepcopy(self.release['payload']);r.update(version=a.GENESIS_RELEASE,method=a.GENESIS_METHOD,genesis_document=self.genesis,genesis_sha256=sha(self.genesis),minimum_optimizer_step=0)
  q=copy.deepcopy(r['execution_qualification']['payload']);q.update(version=a.GENESIS_QUALIFICATION,method=a.GENESIS_METHOD,optimizer_reset=True,tested_methods=[a.GENESIS_METHOD,a.METHOD]);r['execution_qualification']=self.f.sign(q)
  self.genesis_release=self.f.sign(r)
 def bound_epoch(self,n,step):
  job=self.epoch(n,max(1,step),n)
  m=copy.deepcopy(job['manifest']['payload']);b=m['trainer_state_binding'];b['genesis_sha256']=sha(self.genesis);b['global_step_before']=step
  if step==0:b.update(parent=None,genesis=copy.deepcopy(self.genesis))
  else:b['parent']['genesis_sha256']=sha(self.genesis)
  self.rebind(job,m)
  return job
 def rebind(self,job,m):
  epoch=m['epoch'];public=json.loads((self.state/(epoch+'-first-signed-manifest.json')).read_bytes())['payload'];public['trainer_state_binding']=copy.deepcopy(m['trainer_state_binding'])
  (self.state/(epoch+'-first-signed-manifest.json')).write_text(json.dumps(self.f.sign(public)))
  original=copy.deepcopy(m);original.pop('native_training_eligibility_receipt')
  context=self.f.sign(dict(original_signed_manifest=self.f.sign(original),parent_binding_sha256=sha(m['trainer_state_binding'])))
  grades=self.f.sign(dict(context_sha256=sha(context),sampling_assurance='unaudited'))
  subset=self.f.sign(dict(context_sha256=sha(context),grade_receipt_sha256=sha(grades['payload']),accepted_submissions=job['submissions'],accepted_inventory_sha256=sha(learner.receipt_inventory(job['submissions'])),sampling_assurance='unaudited',claims_rewritten=False))
  for name,d in dict(context=context,grades=grades,subset=subset).items():
   (self.state/'native-outcome-eligibility'/epoch/(name+'.ROOT-SIGNED.json')).write_text(json.dumps(d));m['native_training_eligibility_receipt'][name+'_sha256']=sha(d)
  job['manifest']=self.f.sign(m)
 def test_genesis_then_two_successors_automatically_use_single_release(self):
  declarations=[]
  for n,step in ((90,0),(91,1),(92,2)):
   job=self.bound_epoch(n,step);prep=a.automatic_preparation(self.controller,job,self.genesis_release)
   amended=a.attach(job,prep,self.f.authority,self.f.sign);v=a.validate(self.f.sign(amended),self.f.authority)
   declarations.append(v)
   self.assertEqual(v['genesis_sha256'],sha(self.genesis));self.assertEqual(v['execution_release_sha256'],sha(self.genesis_release))
   grant=v['learning_rate_authorization']['payload'];self.assertEqual(grant['optimizer_step_before'],step)
   self.assertEqual(grant['version'],lr.GENESIS_AUTH_VERSION if step==0 else lr.VERSION)
   provenance=a.provenance(self.f.sign(amended),self.f.authority);self.assertEqual(provenance['optimizer_reset'],step==0)
  self.assertEqual([v['version']for v in declarations],[a.GENESIS_VERSION,a.VERSION,a.VERSION])
  self.assertEqual(declarations[0]['learning_rate_authorization']['payload']['run_id'],'7'*64)
 def test_same_genesis_attempt_journal_idempotent(self):
  job=self.bound_epoch(90,0);one=a.automatic_preparation(self.controller,job,self.genesis_release)
  self.assertEqual(one,a.automatic_preparation(self.controller,job,self.genesis_release))
 def test_existing_parent_release_never_authorizes_genesis(self):
  with self.assertRaises(ValueError):a.automatic_preparation(self.controller,self.bound_epoch(90,0),self.release)
 def test_released_genesis_cannot_change_run_id_rate_or_base(self):
  job=self.bound_epoch(90,0)
  for key,value in [('run_id','8'*64),('initial_effective_learning_rate',1e-6),('input_checkpoint','9'*64)]:
   bad=copy.deepcopy(self.genesis_release['payload']);bad['genesis_document'][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):a.automatic_preparation(self.controller,job,self.f.sign(bad))
 def test_qualification_must_prove_initial_and_continuation(self):
  bad=copy.deepcopy(self.genesis_release['payload']);q=copy.deepcopy(bad['execution_qualification']['payload']);q['tested_methods']=[a.GENESIS_METHOD];bad['execution_qualification']=self.f.sign(q)
  with self.assertRaisesRegex(ValueError,'continuation'):a.release(self.f.sign(bad),self.f.authority)
 def test_old_run_successor_cannot_use_new_release(self):
  with self.assertRaises(ValueError):a.automatic_preparation(self.controller,self.epoch(91,2),self.genesis_release)
 def test_signed_genesis_grant_cannot_relabel_as_continuation(self):
  job=self.bound_epoch(90,0);prep=a.automatic_preparation(self.controller,job,self.genesis_release);amended=a.attach(job,prep,self.f.authority,self.f.sign)
  bad=copy.deepcopy(amended);v=copy.deepcopy(bad[a.FIELD]['payload']);v.update(version=a.VERSION,method=a.METHOD);bad[a.FIELD]=self.f.sign(v)
  with self.assertRaises(ValueError):a.validate(self.f.sign(bad),self.f.authority)

if __name__=='__main__':unittest.main()
