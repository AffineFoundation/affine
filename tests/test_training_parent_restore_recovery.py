"""Authenticated pre-update recovery of unchanged unaudited original inputs."""
import copy,json,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from subnet import training_startup_recovery as recovery,committed_training_inputs as learner
from subnet.training_receipts import sha
from subnet.storage import canonical
from subnet.persistent_training_protocol import independently_commit,opening_binding,prepare_job,validate_job

class ParentRestoreRecovery(unittest.TestCase):
 def setUp(self):
  from test_persistent_training_integration import PersistentIntegrationTests
  from test_committed_training_inputs import LearnerAdmissionTests
  fx=PersistentIntegrationTests();fx.setUp();self.addCleanup(fx.doCleanups);self.fx=fx;self.sign=fx.sign;self.authority=fx.authority
  report,parentjob=fx.report();self.pointer=independently_commit(fx.controller,report,parentjob,fx.manifest)
  old=copy.deepcopy(fx.manifest);old['epoch']='next-restore-epoch';old['training_input_policy']=learner.VERSION
  old['trainer_state_binding']=opening_binding(fx.config,dict(round=13,checkpoint=fx.cp,trainer_state=self.pointer),old['epoch'])
  setup=LearnerAdmissionTests();setup.setUp();self.addCleanup(setup.doCleanups)
  setup.operator=fx.key;setup.authority=fx.authority;setup.manifest=old;setup.batch=fx.batch;setup.batch['epoch']=old['epoch']
  for rollout in setup.batch['rollouts']:rollout['environment_version']='synthetic-v1'
  setup.build();self.setup=setup
  self.old=learner.coverage_manifest(old,[setup.obj],seed='f'*64,captured_at=21)
  original=fx.job(self.old,'original-failed-restore',steps=1);original.update(created_at=50,training_input_policy=learner.VERSION)
  original['submissions']=[dict(setup.obj,url=fx.bucket.presign('original-input'))]
  original['source_files'].update({'subnet/'+m+'.py':'e'*64 for m in recovery.SCIENCE})
  original['source_files']['subnet/committed_training_inputs.py']='e'*64
  self.original=original;descriptor=original['persistent_training']['parent_publication']['payload']['descriptor']
  self.terminal=dict(phase='failed',job_id=original['job_id'],exit_code=1,runner_pid=10,runner_pid_ticks='101',child_pid=11,child_pid_ticks='102',started_at=51,finished_at=60)
  witness=dict(version=recovery.RESTORE_WITNESS,observed_at=61,exception='requests.exceptions.ConnectionError',cause='urllib3.exceptions.ReadTimeoutError',failed_stage='parent-state-fetch-before-train_epoch',callchain=['persistent_training_worker.train','persistent_training_state.restore_state','persistent_training_state.restore_one','persistent_training_worker.fetch','persistent_training_worker.cold','backend_jobs.get_object'],model_loaded=True,cuda_allocated=True,original_processes_absent=True,physical_gpu_idle=True,optimizer_step_reached=False,restore_state_returned=False,train_epoch_reached=False,output_checkpoint_absent=True,original_report_absent=True,optimizer_state_candidate_absent=True,update_ledger_absent=True,public_optimizer_steps=3,parent_publication_sha256=sha(original['persistent_training']['parent_publication']),parent_descriptor_sha256=sha(descriptor),parent_shard_count=len(descriptor['shards']),parent_total_bytes=sum(s['size']for s in descriptor['shards']),selected_input_count=1,worker_log_sha256='f'*64,evidence_sha256='f'*64,science_source_files={'subnet/'+m+'.py':'e'*64 for m in recovery.SCIENCE})
  admission=setup.obj['learner_admission']['payload'];self.key='public/'+self.old['epoch']+'/submissions/'+admission['miner_identity']+'/'+admission['commitment_sha256']+'/training/0.json'
  self.value=dict(version=recovery.RESTORE_VERSION,epoch=self.old['epoch'],original_signed_job=self.sign(original),original_job_sha256=sha(original),original_terminal=self.terminal,restore_witness=witness,replacement_source_bundle={'sha256':'f'*64},replacement_job_label='next-restore-recovery',created_at=62,expires_at=200,original_input_source_sha256=self.old['source_bundle']['sha256'],replacement_execution_source_sha256='f'*64,authorized_input_inventory_sha256=sha(recovery.input_inventory(original['submissions'])),authorized_input_objects=[dict(key=self.key,sha256=setup.obj['sha256'],size=setup.obj['size'],learner_admission_sha256=sha(setup.obj['learner_admission']))])
  self.manifest=dict(self.old,source_bundle=self.value['replacement_source_bundle'],training_startup_recovery=self.sign(self.value))
  self.job=copy.deepcopy(original);self.job.update(job_id='next-restore-recovery-fresh',created_at=80,expires_at=100,manifest=self.sign(self.manifest));self.job['source_files']['subnet/training_startup_recovery.py']='1'*64
  self.job['persistent_training']=prepare_job(fx.controller,self.manifest,self.job['job_id'],1,100)
 def changed(self,value=None,job=None,manifest=None):
  m=copy.deepcopy(manifest or self.manifest);m[recovery.FIELD]=self.sign(value or self.value)
  j=copy.deepcopy(job or self.job);j['manifest']=self.sign(m)
  return j,m
 def test_genuine_full_parent_original_unaudited_admission_and_distinct_source(self):
  self.assertEqual(recovery.validate(self.job,self.manifest,self.authority),self.value)
  validate_job(self.job,self.manifest,self.authority);learner.validate_job(self.job,self.manifest,self.authority)
  with patch('subnet.batches.unpack',side_effect=AssertionError('audit')),patch('subnet.model.Runtime.compute',side_effect=AssertionError('model')):
   summary,pairs=learner.admitted_submission(self.setup.path,self.job['submissions'][0],self.manifest,self.authority)
  self.assertEqual(summary['assurance'],'unaudited');self.assertFalse(summary['trainer_verification_performed']);self.assertEqual(len(pairs),1)
  self.assertEqual(self.job['submissions'][0]['learner_admission'],self.original['submissions'][0]['learner_admission'])
  self.assertNotEqual(self.manifest['source_bundle'],self.old['source_bundle']);self.assertEqual(recovery.original_manifest(self.manifest,self.authority),self.old)
 def test_every_ambiguous_or_post_update_witness_rejected(self):
  changes=[(k,not v)for k,v in self.value['restore_witness'].items()if type(v)is bool]
  changes.extend([('callchain',['persistent_training_worker.train','task_normalized_training.train_epoch']),('public_optimizer_steps',4),('parent_shard_count',0),('parent_total_bytes',1),('selected_input_count',2),('worker_log_sha256','bad'),('exception','RuntimeError'),('cause','malformed'),('failed_stage','optimizer-step')])
  for key,value in changes:
   with self.subTest(key=key):
    declaration=copy.deepcopy(self.value);declaration['restore_witness'][key]=value;j,m=self.changed(declaration)
    with self.assertRaises(ValueError):learner.validate_job(j,m,self.authority)
 def test_no_math_parent_input_signature_or_runtime_substitution(self):
  for kind in ('math','parent','input','source','inputscope','runtime','oldid','namespace','expiry','signature','objectkey','objectsize','admissionhash','originalreportphase','originalpid','originalsha'):
   value=copy.deepcopy(self.value);job=copy.deepcopy(self.job);manifest=copy.deepcopy(self.manifest)
   if kind=='math':job['source_files']['subnet/persistent_training_worker.py']='0'*64
   if kind=='parent':manifest['trainer_state_binding']['global_step_before']=0
   if kind=='input':job['submissions'][0]['sha256']='0'*64
   if kind=='source':value['replacement_execution_source_sha256']='0'*64
   if kind=='inputscope':value['original_input_source_sha256']='0'*64
   if kind=='runtime':job['runtime_versions']['torch']='other'
   if kind=='oldid':job['job_id']=self.original['job_id']
   if kind=='namespace':job['persistent_training']['output_namespace']=self.original['persistent_training']['output_namespace']
   if kind=='expiry':job['expires_at']=201
   if kind=='signature':value['original_signed_job']['signature']='bad'
   if kind=='objectkey':value['authorized_input_objects'][0]['key']='private/unrelated.json'
   if kind=='objectsize':value['authorized_input_objects'][0]['size']+=1
   if kind=='admissionhash':value['authorized_input_objects'][0]['learner_admission_sha256']='0'*64
   if kind=='originalreportphase':value['original_terminal']['phase']='complete'
   if kind=='originalpid':value['original_terminal']['runner_pid_ticks']='unknown'
   if kind=='originalsha':value['original_job_sha256']='0'*64
   j,m=self.changed(value,job,manifest)
   with self.subTest(kind=kind),self.assertRaises(Exception):learner.validate_job(j,m,self.authority)
 def test_v1_still_rejects_loaded_model_and_unaudited_recovery(self):
  from test_training_startup_recovery import StartupRecovery
  old=StartupRecovery();old.setUp();old.value['startup_witness']['model_loaded']=True;old.update()
  with self.assertRaisesRegex(ValueError,'pre-compute'):recovery.validate(old.job,old.manifest,old.authority)
  value=copy.deepcopy(self.value);value['version']=recovery.VERSION;j,m=self.changed(value)
  with self.assertRaises(ValueError):learner.validate_job(j,m,self.authority)
 def controller(self):
  fx=self.fx;state=fx.root
  (state/'roles'/(self.old['epoch']+'-train.json')).write_bytes(canonical(dict(job_id=self.original['job_id'],job_sha256=sha(self.original))))
  (state/'roles'/(self.original['job_id']+'-failure.json')).write_bytes(canonical(self.terminal))
  (state/'latest-trainer-state.json').write_bytes(canonical(self.pointer))
  declaration=state/'restore-declaration.json';declaration.write_bytes(canonical(self.sign(self.value)))
  return SimpleNamespace(state=state,authority=SimpleNamespace(id=self.authority),signed=self.sign,bucket=fx.bucket,checkpoint_with_reads=lambda cp:dict(cp,read_urls={'fresh':'GET'}),training_startup_recovery_files={self.old['epoch']:str(declaration)})
 def test_preparation_reuses_frozen_keys_and_reserves_once_preserving_original(self):
  controller=self.controller();originalbytes=(controller.state/'roles'/(self.old['epoch']+'-train.json')).read_bytes()
  with patch('subnet.training_startup_recovery.time.time',return_value=80):manifest,rows=recovery.apply(controller,self.old,1)
  self.assertIn('/'+self.key+'?',rows[0]['url']);self.assertEqual(recovery.input_inventory(rows),recovery.input_inventory(self.original['submissions']))
  self.assertEqual(originalbytes,(controller.state/'roles'/(self.old['epoch']+'-train.json')).read_bytes())
  self.value['replacement_job_label']='attempt-two';Path(controller.training_startup_recovery_files[self.old['epoch']]).write_bytes(canonical(self.sign(self.value)))
  with self.assertRaisesRegex(ValueError,'one immutable'):recovery.apply(controller,self.old,1)
 def test_completed_original_or_changed_public_parent_prevents_reservation(self):
  controller=self.controller();(controller.state/'roles'/(self.original['job_id']+'-report.json')).write_text('{}')
  with self.assertRaisesRegex(ValueError,'original completion'):recovery.apply(controller,self.old,1)
  (controller.state/'roles'/(self.original['job_id']+'-report.json')).unlink();(controller.state/'latest-trainer-state.json').write_text('{}')
  with self.assertRaisesRegex(ValueError,'parent changed'):recovery.apply(controller,self.old,1)
  self.assertFalse((controller.state/(self.old['epoch']+'-startup-recovery-reservation.json')).exists())

 def test_backend_constructor_and_receipt_check_before_new_transport(self):
  from subnet.backend_jobs import validate
  with patch('subnet.backend_jobs.time.time',return_value=80):validate(self.sign(self.job),self.authority,now=80)
  # RemoteJobs authenticates receipt inputs before preparing new persistent
  # capabilities. The old signed publication remains available in declaration.
  early=copy.deepcopy(self.job);early.pop('persistent_training')
  learner.validate_job(early,self.manifest,self.authority)
 def test_controller_dispatch_uses_new_label_and_original_population_without_audit(self):
  from subnet.persistent_training_controller import train
  controller=self.controller();state=controller.state
  (state/(self.old['epoch']+'-learner-population.json')).write_bytes(canonical(dict(version=learner.VERSION,manifest=self.old,submissions=self.original['submissions'],population={'assurance':'unaudited'})))
  seen=[]
  def probe(manifest,steps,*,submission_bytes):
   seen.append((manifest,steps,submission_bytes));raise RuntimeError('stop-before-dispatch')
  controller.jobs=SimpleNamespace(persistent_training_capacity=probe)
  with patch('subnet.training_startup_recovery.time.time',return_value=80),patch('subnet.training_receipts.prepare_submissions',side_effect=AssertionError('audit')),self.assertRaisesRegex(RuntimeError,'stop-before-dispatch'):
   train(controller,self.old,{},'/unused',steps=1)
  self.assertEqual(len(seen),1);self.assertEqual(seen[0][0]['source_bundle'],self.value['replacement_source_bundle'])
  self.assertEqual(recovery.label(controller,self.old['epoch']),self.value['replacement_job_label'])
  self.assertEqual(seen[0][2],self.original['submissions'][0]['size'])
 def test_actual_ten_scientific_files_required_no_invented_sampling_module(self):
  value=copy.deepcopy(self.value);original=copy.deepcopy(self.original);original['source_files'].pop('subnet/sampling_contract.py')
  value['original_signed_job']=self.sign(original);value['original_job_sha256']=sha(original);value['restore_witness']['science_source_files'].pop('subnet/sampling_contract.py')
  job=copy.deepcopy(self.job);job['source_files'].pop('subnet/sampling_contract.py');j,m=self.changed(value,job)
  learner.validate_job(j,m,self.authority)
  original['source_files'].pop('subnet/task_normalized_training.py');value['original_signed_job']=self.sign(original);value['original_job_sha256']=sha(original);value['restore_witness']['science_source_files'].pop('subnet/task_normalized_training.py');j,m=self.changed(value,job)
  with self.assertRaisesRegex(ValueError,'source pins'):learner.validate_job(j,m,self.authority)
