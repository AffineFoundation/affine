"""A failed export is not a zero-update failure or a durable new parent."""
import copy,unittest
from unittest.mock import patch
from subnet import training_startup_recovery as r,committed_training_inputs as learner
from subnet.training_receipts import sha
from subnet.storage import canonical
from test_training_parent_restore_recovery import ParentRestoreRecovery

class PostUpdateRecovery(ParentRestoreRecovery):
 def setUp(self):
  super().setUp()
  w=self.value.pop('restore_witness')
  for k in ('cause','model_loaded','cuda_allocated','restore_state_returned','train_epoch_reached','output_checkpoint_absent','optimizer_state_candidate_absent','update_ledger_absent'):w.pop(k)
  w.update(version=r.POST_UPDATE_WITNESS,exception='ValueError: persistent state PUT status',failed_stage='post-update-persistent-state-export',callchain=['persistent_training_worker.train','persistent_training_state.export_state','persistent_training_state._export_state','persistent_training_state.transfer_one','persistent_training_state.materialize','persistent_training_worker.publish','persistent_training_worker.put_file'],optimizer_step_reached=True,optimizer_updates=1,train_epoch_returned=True,complete_candidate_descriptor_absent=True,candidate_publication_absent=True,failed_output_namespace=self.original['persistent_training']['output_namespace'],partial_inventory_sha256='8'*64,uploaded_shard_count=13,local_shard_count=14,planned_shard_count=23)
  self.value.update(version=r.POST_UPDATE_VERSION,post_update_witness=w,execution_science_source_files=copy.deepcopy(w['science_source_files']),execution_runtime_source_files=copy.deepcopy(self.job['source_files']),approval=dict(fresh_attempt_from_last_durable_parent=True,original_update_occurred=True,incomplete_candidate_is_not_parent=True,preserve_original_failure=True,maximum_fresh_attempts=1))
  self.job,self.manifest=self.changed()
 def test_explicit_postupdate_new_attempt_and_original_admissions(self):
  self.assertEqual(r.validate(self.job,self.manifest,self.authority),self.value)
  learner.validate_job(self.job,self.manifest,self.authority)
  self.assertEqual(r.original_manifest(self.manifest,self.authority),self.old)
 def test_zero_update_or_published_candidate_or_false_approval_refused(self):
  mutations=[('optimizer_updates',0),('optimizer_updates',True),('optimizer_step_reached',False),('original_processes_absent',False),('physical_gpu_idle',False),('complete_candidate_descriptor_absent',False),('candidate_publication_absent',False),('original_report_absent',False),('uploaded_shard_count',23),('local_shard_count',23),('failed_output_namespace','unowned'),('partial_inventory_sha256','bad'),('public_optimizer_steps',4),('selected_input_count',2),('exception','HTTP409')]
  for key,value in mutations:
   d=copy.deepcopy(self.value);d['post_update_witness'][key]=value;j,m=self.changed(d)
   with self.subTest(key=key),self.assertRaises(ValueError):r.validate(j,m,self.authority)
  d=copy.deepcopy(self.value);d['approval']['maximum_fresh_attempts']=2;j,m=self.changed(d)
  with self.assertRaises(ValueError):r.validate(j,m,self.authority)
 def test_exact_approved_transport_delta_only_no_math_change(self):
  for name in r.POST_UPDATE_OPERATIONAL_MODULES:
   d=copy.deepcopy(self.value);d['execution_science_source_files'][name]='a'*64;d['execution_runtime_source_files'][name]='a'*64;j=copy.deepcopy(self.job);j['source_files'][name]='a'*64;j,m=self.changed(d,j)
   learner.validate_job(j,m,self.authority)
  for name in self.value['execution_science_source_files']:
   if name in r.POST_UPDATE_OPERATIONAL_MODULES:continue
   d=copy.deepcopy(self.value);d['execution_science_source_files'][name]='a'*64;d['execution_runtime_source_files'][name]='a'*64;j=copy.deepcopy(self.job);j['source_files'][name]='a'*64;j,m=self.changed(d,j)
   with self.subTest(name=name),self.assertRaisesRegex(ValueError,'mathematical'):r.validate(j,m,self.authority)
 def test_distinct_one_reservation_never_overwrites_preupdate_history(self):
  c=self.controller();old=r.reservation_path(c.state,self.old['epoch'],r.RESTORE_VERSION);old.write_bytes(b'preserved-history')
  with patch('subnet.training_startup_recovery.time.time',return_value=80):r.apply(c,self.old,1)
  self.assertEqual(old.read_bytes(),b'preserved-history');self.assertTrue(r.reservation_path(c.state,self.old['epoch'],r.POST_UPDATE_VERSION).exists())
 def test_local_evidence_reports_one_uncommitted_update_honestly(self):
  c=self.controller()
  with patch('subnet.training_startup_recovery.time.time',return_value=80):r.apply(c,self.old,1)
  label=self.value['replacement_job_label'];record=dict(job_id=self.job['job_id'],job_sha256=sha(self.job))
  (c.state/'roles'/(label+'.json')).write_bytes(canonical(record));(c.state/'roles'/(self.job['job_id']+'-job.json')).write_bytes(canonical(self.sign(self.job)))
  _,_,e=r.local_request(c.state,self.old['epoch'],self.authority)
  self.assertEqual(e['original_optimizer_updates'],1);self.assertTrue(e['original_update_uncommitted']);self.assertTrue(e['restarted_from_durable_parent']);self.assertEqual(e['original_failed_stage'],'post-update-persistent-state-export')
 # Inherited test cases using pre-update-specific mutations are exercised by
 # their original class; these two are intentionally not this new protocol.
 test_every_ambiguous_or_post_update_witness_rejected=None
 test_v1_still_rejects_loaded_model_and_unaudited_recovery=None
 test_actual_ten_scientific_files_required_no_invented_sampling_module=None
