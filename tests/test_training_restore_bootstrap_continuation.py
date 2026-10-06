"""One small authenticated v3 sibling; preserve both original failed attempts."""
import copy,hashlib,json,unittest
from unittest.mock import patch
from subnet import training_startup_recovery as r,committed_training_inputs as learner
from subnet.storage import canonical
from subnet.training_receipts import sha
from subnet.persistent_training_protocol import prepare_job
from test_training_parent_restore_recovery import ParentRestoreRecovery

class BootstrapContinuation(unittest.TestCase):
 def setUp(self):
  fx=ParentRestoreRecovery();fx.setUp();self.addCleanup(fx.doCleanups);self.fx=fx;self.sign=fx.sign;self.authority=fx.authority;self.controller=fx.controller();self.state=self.controller.state
  with patch('subnet.training_startup_recovery.time.time',return_value=80):prior_manifest,prior_rows=r.apply(self.controller,fx.old,1)
  self.v2path=r.reservation_path(self.state,fx.old['epoch'],r.RESTORE_VERSION);self.v2raw=self.v2path.read_bytes();journal=json.loads(self.v2raw)
  self.prior=copy.deepcopy(fx.job);self.prior.update(manifest=self.sign(prior_manifest),submissions=prior_rows)
  self.prior_record=dict(job_id=self.prior['job_id'],job_sha256=sha(self.prior));self.recordpath=self.state/'roles'/(fx.value['replacement_job_label']+'.json');self.recordpath.write_bytes(canonical(self.prior_record));self.jobpath=self.state/'roles'/(self.prior['job_id']+'-job.json');self.jobpath.write_bytes(canonical(self.sign(self.prior)))
  terminal=dict(phase='failed',job_id=self.prior['job_id'],exit_code=1,runner_pid=20,runner_pid_ticks='201',child_pid=21,child_pid_ticks='202',started_at=85,finished_at=86);self.failurepath=self.state/'roles'/(self.prior['job_id']+'-failure.json');self.failurepath.write_bytes(canonical(terminal))
  witness=dict(version='operator-recovery-envelope-guard-witness-v1',observed_at=87,exception='ValueError: job envelope size budget',failed_stage='backend_jobs.main-before-execute',cuda_allocated=False,model_loaded=False,execution_started=False,original_processes_absent=True,output_namespace_empty=True,physical_gpu_idle=True,worker_log_sha256='a'*64,worker_log_bytes=633,evidence_sha256='b'*64)
  self.value=copy.deepcopy(fx.value);self.value.update(version=r.BOOTSTRAP_VERSION,replacement_source_bundle={'sha256':'d'*64},replacement_execution_source_sha256='d'*64,replacement_job_label='next-restore-bootstrap-v3',created_at=88,predecessor=dict(version='terminal-recovery-envelope-guard-predecessor-v1',job_id=self.prior['job_id'],job_sha256=sha(self.prior),execution_source_sha256=prior_manifest['source_bundle']['sha256'],reservation_sha256=hashlib.sha256(self.v2raw).hexdigest(),declaration_sha256=journal['declaration_sha256'],input_inventory_sha256=sha(r.input_inventory(self.prior['submissions'])),terminal=terminal,bootstrap_witness=witness))
  self.manifest=dict(fx.old,source_bundle=self.value['replacement_source_bundle'],training_startup_recovery=self.sign(self.value));self.job=copy.deepcopy(fx.job);self.job.update(job_id=self.value['replacement_job_label']+'-original',created_at=90,manifest=self.sign(self.manifest));self.job['persistent_training']=prepare_job(fx.fx.controller,self.manifest,self.job['job_id'],1,100)
  self.document=self.state/'bootstrap-v3-declaration.json';self.document.write_bytes(canonical(self.sign(self.value)));self.controller.training_startup_recovery_files={fx.old['epoch']:str(self.document)}
 def test_one_small_sibling_authenticates_original_inputs_science_and_both_failures(self):
  learner.validate_job(self.job,self.manifest,self.authority);self.assertEqual(r.validate_predecessor_local(self.state,self.value,self.authority),self.prior)
  self.assertNotIn('original_signed_job',self.value['predecessor']);self.assertLess(len(canonical(self.value['predecessor'])),2500)
  with patch('subnet.training_startup_recovery.time.time',return_value=90):manifest,rows=r.apply(self.controller,self.fx.old,1)
  self.assertEqual(self.v2path.read_bytes(),self.v2raw);self.assertTrue(r.reservation_path(self.state,self.fx.old['epoch'],r.BOOTSTRAP_VERSION).exists());self.assertEqual(r.input_inventory(rows),r.input_inventory(self.fx.original['submissions']));self.assertNotEqual(manifest['source_bundle'],self.fx.old['source_bundle'])
 def test_unknown_live_postcompute_or_changed_predecessor_witness_rejected(self):
  changes=[('cuda_allocated',True),('model_loaded',True),('execution_started',True),('original_processes_absent',False),('physical_gpu_idle',False),('output_namespace_empty',False),('exception','timeout'),('worker_log_bytes',0),('worker_log_sha256','bad'),('failed_stage','train_epoch'),('observed_at',201)]
  for key,value in changes:
   v=copy.deepcopy(self.value);v['predecessor']['bootstrap_witness'][key]=value;m=dict(self.manifest,training_startup_recovery=self.sign(v))
   with self.subTest(key=key),self.assertRaises(ValueError):learner.validate_job(self.job,m,self.authority)
 def test_any_previous_journal_request_terminal_or_completion_tamper_holds(self):
  for kind in ('journalbytes','jobdigest','failure','report','signature','v3chain','missing'):
   with self.subTest(kind=kind):
    oldjobraw=self.jobpath.read_bytes();oldfailraw=self.failurepath.read_bytes();oldrecordraw=self.recordpath.read_bytes()
    report=self.state/'roles'/(self.prior['job_id']+'-report.json')
    if kind=='journalbytes':self.v2path.write_bytes(self.v2raw+b' ')
    if kind=='jobdigest':self.recordpath.write_bytes(canonical(dict(self.prior_record,job_sha256='0'*64)))
    if kind=='failure':self.failurepath.write_text('{}')
    if kind=='report':report.write_text('{}')
    if kind=='signature':job=json.loads(oldjobraw);job['signature']='bad';self.jobpath.write_bytes(canonical(job))
    if kind=='v3chain':j=json.loads(self.v2raw);v=j['declaration']['payload'];v['version']=r.BOOTSTRAP_VERSION;j['declaration']=self.sign(v);self.v2path.write_bytes(canonical(j))
    if kind=='missing':self.v2path.unlink()
    with self.assertRaises(Exception):r.apply(self.controller,self.fx.old,1)
    self.assertFalse(r.reservation_path(self.state,self.fx.old['epoch'],r.BOOTSTRAP_VERSION).exists())
    self.v2path.write_bytes(self.v2raw);self.jobpath.write_bytes(oldjobraw);self.failurepath.write_bytes(oldfailraw);self.recordpath.write_bytes(oldrecordraw);report.unlink(missing_ok=True)
 def test_new_reservation_cannot_be_replaced_or_repeat_v3(self):
  with patch('subnet.training_startup_recovery.time.time',return_value=90):r.apply(self.controller,self.fx.old,1)
  value=copy.deepcopy(self.value);value['replacement_job_label']='repeat-v3';self.document.write_bytes(canonical(self.sign(value)))
  with self.assertRaisesRegex(ValueError,'one immutable'):r.apply(self.controller,self.fx.old,1)
  self.assertEqual(self.v2path.read_bytes(),self.v2raw)
 def test_local_completion_selects_only_valid_v3_and_preserves_input_execution_attribution(self):
  with patch('subnet.training_startup_recovery.time.time',return_value=90):manifest,rows=r.apply(self.controller,self.fx.old,1)
  job=copy.deepcopy(self.job);job.update(manifest=self.sign(manifest),submissions=rows)
  record=dict(job_id=job['job_id'],job_sha256=sha(job));(self.state/'roles'/(self.value['replacement_job_label']+'.json')).write_bytes(canonical(record));(self.state/'roles'/(job['job_id']+'-job.json')).write_bytes(canonical(self.sign(job)))
  selected,request,evidence=r.local_request(self.state,self.fx.old['epoch'],self.authority)
  self.assertEqual(selected,record);self.assertEqual(request,job);self.assertEqual(evidence['bootstrap_predecessor'],self.value['predecessor']);self.assertEqual(evidence['original_input_source_sha256'],self.fx.old['source_bundle']['sha256']);self.assertEqual(evidence['replacement_execution_source_sha256'],'d'*64)
  self.v2path.write_bytes(self.v2raw+b' ')
  with self.assertRaises(ValueError):r.local_request(self.state,self.fx.old['epoch'],self.authority)

 def test_v2_journal_cannot_be_copied_to_v3_slot_to_select_prior_failed_job(self):
  r.reservation_path(self.state,self.fx.old['epoch'],r.BOOTSTRAP_VERSION).write_bytes(self.v2raw)
  with self.assertRaisesRegex(ValueError,'explicit v3'):r.local_request(self.state,self.fx.old['epoch'],self.authority)
