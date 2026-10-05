import copy,json,unittest
from types import SimpleNamespace
from subnet.persistent_cpu_adamw import sha
from subnet.remote_backend import RemoteJobs
from subnet.training_quality_adapter import training_input,evaluation_schedule
from test_persistent_training_integration import PersistentIntegrationTests

class Adapter(unittest.TestCase):
 def setUp(self):
  self.fixture=PersistentIntegrationTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
  self.report,self.job=self.fixture.report();self.manifest=self.job['manifest']['payload'];self.authority=self.fixture.authority
  self.report.update(execution_runtime_revision=self.manifest['model_runtime_revision'],generation_runtime_revision=self.manifest['model_runtime_revision'])
  self.prior=dict(job_id=self.job['job_id'],role='train',job_sha256=sha(self.job),source_files=self.job['source_files'],runtime_versions=self.job['runtime_versions'],manifest_sha256=sha(self.manifest))
  self.jobs=RemoteJobs.__new__(RemoteJobs);self.jobs.state=self.fixture.root/'roles';self.jobs.controller=self.fixture.controller
  (self.jobs.state/(self.job['job_id']+'-job.json')).write_text(json.dumps(self.fixture.sign(self.job)))
  from subnet.persistent_training_protocol import PUBLICATION_VERSION
  state=self.report['persistent_training_state'];self.publication=self.fixture.sign(dict(version=PUBLICATION_VERSION,namespace=state['namespace'],job_id=self.job['job_id'],job_sha256=sha(self.job),descriptor_sha256=state['descriptor_sha256'],descriptor=state['descriptor']))
 def run_adapter(self,report=None,publication=True):
  return training_input(self.jobs,self.prior,report or self.report,self.fixture.sign(self.job),self.publication if publication else None,self.authority)
 def test_actual_tiny_optimizer_checked_report_adoption(self):
  value=self.run_adapter();self.assertEqual(value['optimizer_step_after'],3);self.assertFalse(value['inference_weights_changed']);self.assertFalse(value['convergence_claimed'])
 def test_uncommitted_is_pending(self):self.assertEqual(self.run_adapter(publication=False)['status'],'pending_durable_adoption')
 def test_wrong_report_runtime_source_precision_rejected(self):
  for field,value in [('job_id','foreign'),('runtime_versions',{}),('backend_profile',{}),('source_files',{}),('checkpoint','ff'*32)]:
   row=copy.deepcopy(self.report);row[field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):self.run_adapter(report=row)
  row=copy.deepcopy(self.report);row['training']['updates'][0]['precision']['bf16_changed_elements']=999
  with self.assertRaises(ValueError):self.run_adapter(report=row)
 def test_wrong_state_branch_counter_signature(self):
  for field,value in [('job_sha256','ff'*32),('namespace','foreign')]:
   old=self.publication;payload=copy.deepcopy(old['payload']);payload[field]=value;self.publication=self.fixture.sign(payload)
   with self.assertRaises(ValueError):self.run_adapter()
   self.publication=old
  self.publication['payload']['descriptor_sha256']='ff'*32
  with self.assertRaises(Exception):self.run_adapter()
 def test_finite_loss_margin_original_validation(self):
  row=copy.deepcopy(self.report);row['training']['updates'][0]['loss']=float('nan')
  with self.assertRaises(ValueError):self.run_adapter(report=row)
 def test_128_rotation_full750_existing_bounded_chunks(self):
  reserved=list(range(750));mining=list(range(750,900))
  self.assertEqual([len(c)for c in evaluation_schedule(reserved,mining,0)['chunks']],[64,64])
  full=evaluation_schedule(reserved,mining,23);self.assertEqual(len(full['indices']),750);self.assertEqual(len(full['chunks']),12);self.assertFalse(full['training_barrier'])

class EvaluationAdapter(unittest.TestCase):
 def setUp(self):
  from test_remote_backend import RemoteReportBinding
  from subnet.backend_profiles import resolve
  self.f=RemoteReportBinding();self.f.setUp();self.addCleanup(self.f.doCleanups)
  self.jobs=self.f.jobs;self.authority=self.f.operator;self.sign=lambda value:__import__('subnet.remote_optimizer_readback',fromlist=['sign']).sign(value,self.f.key)
  self.manifest=dict(self.f.manifest,environments=[dict(env_id='math',spec={'version':'v1'},harness={})],harness_source_hash='aa'*32)
  self.suite=dict(env_id='math',indices=[7],seeds=[99],harness={})
  self.job=dict(self.f.envelope['payload'],role='evaluate',manifest=self.sign(self.manifest),heldout=[self.suite])
  self.envelope=self.sign(self.job)
  self.prior=dict(self.f.prior,role='evaluate',job_sha256=sha(self.job),manifest_sha256=sha(self.manifest))
  source={n:'ab'*32 for n in ('subnet/model.py','subnet/gpu_runtime.py','subnet/environments.py','subnet/harness.py','subnet/proofs.py')}
  self.prior['source_files']=source
  self.report=dict(self.f.report,role='evaluate',job_sha256=sha(self.job),source_files=source,
   heldout=[dict(env_id='math',index=7,seed=99,verified=True,classification='positive',reward=1,task_hash='cc'*32)])
  (self.jobs.state/'same-job-job.json').write_text(json.dumps(self.envelope))
  revision,profile,_=resolve(self.manifest)
  frozen=dict(env_id='math',environment={'version':'v1'},harness={},indices=[7],seeds=[99],model_runtime_revision=revision,backend_profile=profile,runtime_versions=self.report['runtime_versions'],harness_source_hash='aa'*32,source_files=source)
  record=dict(env_id='math',checkpoint='approved',remote_job_id='same-job',timestamp=20,heldout_indices=[7],harness_config={},model_runtime_revision=revision,backend_profile=profile,dataset_id=sha(frozen),taskset_hash=sha(frozen),requested_count=1,completed_count=1,successes=1,fixed_task_ids=['cc'*32],task_hashes=['cc'*32],evaluation_failures=[])
  self.export=dict(epoch='nonpayable-test',checkpoint='approved',phase='before',status='complete',public_optimizer_steps=4,records=[record])
 def run_adapter(self,document=None):
  from subnet.training_quality_adapter import evaluation_input
  return evaluation_input(self.jobs,self.prior,self.report,self.envelope,document or self.sign(self.export),self.authority)
 def test_original_checked_eval_export(self):self.assertEqual(self.run_adapter()['records'][0]['successes'],1)
 def test_wrong_task_hash_seed_branch_runtime_signature(self):
  for field,value in [('fixed_task_ids',['dd'*32]),('heldout_indices',[8]),('backend_profile',{}),('dataset_id','ff'*32)]:
   export=copy.deepcopy(self.export);export['records'][0][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):self.run_adapter(self.sign(export))
  export=copy.deepcopy(self.export);export['checkpoint']='different'
  with self.assertRaises(ValueError):self.run_adapter(self.sign(export))
  doc=self.sign(self.export);doc['payload']['public_optimizer_steps']=5
  with self.assertRaises(Exception):self.run_adapter(doc)
 def test_aggregate_chunks_exact_cohort_not_two_small_confirmations(self):
  from subnet.training_quality_adapter import aggregate_evaluations
  first=self.run_adapter();second=copy.deepcopy(first);second['records'][0].update(heldout_indices=[8],fixed_task_ids=['dd'*32],task_hashes=['dd'*32],dataset_id='bb'*32)
  documents=[self.sign(first),self.sign(second)]
  row=aggregate_evaluations(documents,self.authority,checkpoint='approved',optimizer_steps=4,expected_indices=[7,8])
  self.assertEqual(len(row['records']),1);self.assertEqual(row['records'][0]['requested_count'],2)
  self.assertEqual(aggregate_evaluations(documents[:1],self.authority,checkpoint='approved',optimizer_steps=4,expected_indices=[7,8])['status'],'pending_paired_evidence')
  with self.assertRaises(ValueError):aggregate_evaluations([documents[0],documents[0]],self.authority,checkpoint='approved',optimizer_steps=4,expected_indices=[7,8])
 def test_comparison_labels_preserve_original_execution(self):
  from subnet.training_quality_adapter import comparison_views
  before=self.run_adapter();after=copy.deepcopy(before);after.update(epoch='later-epoch',phase='after',checkpoint='new')
  training=dict(version='authenticated-training-quality-input-v1',status='complete',epoch='transition',input_checkpoint='approved',output_checkpoint='new')
  result=comparison_views(self.sign(before),self.sign(after),self.sign(training),self.authority)
  self.assertEqual(result['before']['epoch'],'transition');self.assertEqual(result['before']['comparison_provenance']['original_epoch'],'nonpayable-test')
  training['input_checkpoint']='foreign'
  with self.assertRaises(ValueError):comparison_views(self.sign(before),self.sign(after),self.sign(training),self.authority)
