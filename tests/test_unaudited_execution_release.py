"""Synthetic ROOT release automatically authorizes consecutive native epochs."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from nacl.signing import SigningKey
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'tests')]
from test_unaudited_execution_contract import AmendmentTests
from training_receipt_fixtures import sign
from subnet import unaudited_training_execution as a
from subnet import committed_training_inputs as learner
from subnet.training_receipts import sha

class ReleaseTests(unittest.TestCase):
 def setUp(self):
  self.f=AmendmentTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.state=Path(self.temp.name)
  self.controller=SimpleNamespace(state=self.state,authority=SimpleNamespace(id=self.f.authority),signed=self.f.sign)
  v=self.f.value;r={k:copy.deepcopy(v[k])for k in a.RELEASE_FIELDS if k in v}
  r.update(version=a.RELEASE_VERSION,created_at=20,expires_at=10_000,epoch_prefix='nonpayable-synthetic--',first_round=90,minimum_optimizer_step=2,hyperparameters_sha256=sha(self.f.manifest['payload']['trainer_state_binding']['hyperparameters']))
  self.release=self.f.sign(r)
 def epoch(self,n=90,step=2,index=1):
  m=copy.deepcopy(self.f.manifest['payload']);m['epoch']='nonpayable-synthetic--1000-'+str(n)
  binding=m['trainer_state_binding'];binding['epoch']=m['epoch'];binding['global_step_before']=step;binding['parent']['optimizer_steps']=step;binding['parent']['descriptor_sha256']=sha({'step':step})
  m.pop('native_training_eligibility_receipt',None)
  obj=copy.deepcopy(self.f.submissions[0]);admission=obj['learner_admission']['payload'];miner=SigningKey.generate();identity=miner.verify_key.encode().hex()
  c=admission['original_commitment']['payload'];c['epoch']=m['epoch'];c['miner']=identity;c['batches'][0]['index']=index
  original=sign(miner,c);admission.update(epoch=m['epoch'],miner_identity=identity,original_commitment=original,commitment_sha256=sha(original));obj['learner_admission']=self.f.sign(admission)
  public=self.f.sign(m);m=learner.coverage_manifest(m,[obj],seed='5'*64,captured_at=21)
  context=self.f.sign(dict(original_signed_manifest=self.f.sign(m),parent_binding_sha256=sha(binding)))
  grades=self.f.sign(dict(context_sha256=sha(context),sampling_assurance='unaudited'))
  subset=self.f.sign(dict(context_sha256=sha(context),grade_receipt_sha256=sha(grades['payload']),accepted_submissions=[obj],accepted_inventory_sha256=sha(learner.receipt_inventory([obj])),sampling_assurance='unaudited',claims_rewritten=False))
  docs=dict(context=context,grades=grades,subset=subset)
  m['native_training_eligibility_receipt']=dict(version='native-outcome-accepted-subset-v1',context_sha256=sha(context),grades_sha256=sha(grades),subset_sha256=sha(subset),authorization_sha256='6'*64,sampling_assurance='unaudited',proof_verification_performed=False,claims_rewritten=False,cheating_penalties=False)
  job=copy.deepcopy(self.f.job);job.update(job_id='automatic-train-'+str(n),manifest=self.f.sign(m),submissions=[obj],created_at=30+n,expires_at=130+n)
  (self.state/(m['epoch']+'-first-signed-manifest.json')).write_text(json.dumps(public))
  root=self.state/'native-outcome-eligibility'/m['epoch'];root.mkdir(parents=True)
  for key,d in docs.items():(root/(key+'.ROOT-SIGNED.json')).write_text(json.dumps(d))
  return job
 def test_two_consecutive_epochs_need_one_release_and_distinct_scoped_grants(self):
  j1=self.epoch(90,2,1);g1=a.automatic_preparation(self.controller,j1,self.release);o1=a.attach(j1,g1,self.f.authority,self.f.sign)
  j2=self.epoch(91,3,2);g2=a.automatic_preparation(self.controller,j2,self.release);o2=a.attach(j2,g2,self.f.authority,self.f.sign)
  v1=a.validate(self.f.sign(o1),self.f.authority);v2=a.validate(self.f.sign(o2),self.f.authority)
  self.assertEqual(v1['execution_release_sha256'],v2['execution_release_sha256'])
  for key in('epoch','input_inventory_sha256','trainer_binding_sha256','parent_descriptor_sha256','optimizer_step_before'):
   self.assertNotEqual(v1[key],v2[key])
  self.assertEqual(len(list((self.state/'roles/execution-preparations').glob('*.json'))),2)
 def test_repeated_original_job_reuses_exact_immutable_preparation(self):
  job=self.epoch();first=a.automatic_preparation(self.controller,job,self.release)
  path=next((self.state/'roles/execution-preparations').glob('*.json'));original=path.read_bytes()
  later=copy.deepcopy(job);later['created_at']+=10;later['expires_at']+=10
  again=a.automatic_preparation(self.controller,later,self.release)
  self.assertEqual(first,again);self.assertEqual(original,path.read_bytes())
  self.assertEqual(a.attach(job,first,self.f.authority,self.f.sign),a.attach(job,again,self.f.authority,self.f.sign))
 def test_bad_release_signature_or_candidate_hash_refused(self):
  job=self.epoch();bad=copy.deepcopy(self.release);bad['payload']['training_source_bundle']['sha256']='0'*64
  for grant in(bad,self.f.sign(bad['payload'])):
   with self.assertRaises(ValueError):a.automatic_preparation(self.controller,job,grant)
 def test_genesis_and_hyperparameters_cannot_change_under_release(self):
  job=self.epoch()
  for key in('genesis_sha256','hyperparameters_sha256'):
   release=copy.deepcopy(self.release['payload']);release[key]='0'*64
   with self.subTest(key=key),self.assertRaises(ValueError):a.automatic_preparation(self.controller,job,self.f.sign(release))
 def test_activation_boundary_rejects_prior_round(self):
  with self.assertRaisesRegex(ValueError,'scope'):a.automatic_preparation(self.controller,self.epoch(89),self.release)
 def test_tampered_native_context_and_subset_refused(self):
  job=self.epoch();root=self.state/'native-outcome-eligibility'/job['manifest']['payload']['epoch']
  p=root/'subset.ROOT-SIGNED.json';d=json.loads(p.read_bytes());d['payload']['accepted_submissions']=[];p.write_text(json.dumps(d))
  with self.assertRaises(ValueError):a.automatic_preparation(self.controller,job,self.release)
 def test_changed_inputs_cannot_reuse_same_epoch_preparation(self):
  job=self.epoch();a.automatic_preparation(self.controller,job,self.release);job['submissions']=[]
  with self.assertRaises(ValueError):a.automatic_preparation(self.controller,job,self.release)
 def test_unauthorized_source_or_runtime_refused_before_signature(self):
  job=self.epoch()
  for key in('source_files','runtime_versions'):
   changed=copy.deepcopy(job);changed[key]={}
   with self.subTest(key=key),self.assertRaises(ValueError):a.automatic_preparation(self.controller,changed,self.release)
 def test_released_qualification_still_requires_actual_GPU_evidence(self):
  job=self.epoch();r=copy.deepcopy(self.release['payload']);q=copy.deepcopy(r['execution_qualification']['payload']);q['actual_GPU_execution']=False;r['execution_qualification']=self.f.sign(q)
  with self.assertRaisesRegex(ValueError,'qualification'):a.automatic_preparation(self.controller,job,self.f.sign(r))
 def test_grant_journal_contains_no_new_manifest_copy(self):
  job=self.epoch();value=a.automatic_preparation(self.controller,job,self.release)['payload']
  self.assertNotIn('manifest',value);self.assertNotIn('original_manifest',value)
  self.assertEqual(value['original_signed_manifest_sha256'],sha(job['manifest']))


class AutomaticRemoteDispatchTests(ReleaseTests):
 def test_actual_RemoteJobs_derives_both_epochs_without_new_configuration(self):
  from unittest.mock import Mock,patch
  from subnet.remote_backend import RemoteJobs
  jobs=RemoteJobs.__new__(RemoteJobs);jobs.state=self.state/'roles';jobs.state.mkdir();jobs.workspace='/synthetic-qualified-trainer';jobs.controller=self.controller
  jobs.metadata={k:copy.deepcopy(self.f.job[k])for k in('source_files','runtime_versions')}
  jobs.config={'unaudited_training_execution_release':self.release};jobs.command=Mock();jobs.copy_to=Mock();jobs.launch_runner=Mock();jobs.remote_status=Mock(return_value={'phase':'complete'});jobs.checked=Mock(return_value={'synthetic_only':True});jobs.copy_from=lambda remote,path:path.write_text('{}')
  declarations=[]
  for n,step in((90,2),(91,3)):
   job=self.epoch(n,step,n)
   with patch('subnet.remote_backend.time.time',return_value=job['created_at']),patch('subnet.remote_backend.secrets.token_hex',return_value='abcdef00'),patch('subnet.persistent_training_protocol.prepare_job',return_value={'synthetic_only':True}),patch('subnet.persistent_training_protocol.validate_job'):
    jobs.run('train'+str(n),'train',job['manifest']['payload'],submissions=job['submissions'],steps=1,training_policy=job['training_policy'])
   env=json.loads((jobs.state/('train'+str(n)+'-abcdef00-job.json')).read_bytes());declarations.append(a.validate(env,self.f.authority))
  self.assertEqual(jobs.launch_runner.call_count,2)
  self.assertEqual(declarations[0]['execution_release_sha256'],declarations[1]['execution_release_sha256'])
  self.assertNotEqual(declarations[0]['input_inventory_sha256'],declarations[1]['input_inventory_sha256'])
  self.assertEqual([x['optimizer_step_before']for x in declarations],[2,3])

if __name__=='__main__':unittest.main()
