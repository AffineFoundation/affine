import copy
from test_training_parent_restore_recovery import ParentRestoreRecovery
from subnet import training_startup_recovery as r

class CacheAckRecovery(ParentRestoreRecovery):
 def setUp(self):
  super().setUp()
  self.value['version']=r.CACHE_ACK_VERSION
  self.value['restore_witness'].update(version=r.CACHE_ACK_WITNESS,exception='ValueError',cause='approved parent candidate awaits original durability ACK; refuse cold abandonment',failed_stage='parent-cache-ACK-before-restore_state',callchain=['persistent_training_worker.train','optimizer_state_cache.prepare_parent'])
  self.original['source_files']['subnet/training_startup_recovery.py']='e'*64
  self.value['original_signed_job']=self.sign(self.original)
  self.value['original_job_sha256']=r.sha(self.original)
  self.value['execution_runtime_source_files']=copy.deepcopy(self.job['source_files'])
  self.job,self.manifest=self.changed()
 def test_other_runtime_drift_rejected(self):
  for name in ('subnet/model.py','subnet/persistent_cpu_adamw.py','subnet/forced_sampling.py','subnet/persistent_training_worker.py','subnet/committed_training_inputs.py'):
   value=copy.deepcopy(self.value);job=copy.deepcopy(self.job)
   value['execution_runtime_source_files'][name]='9'*64;job['source_files'][name]='9'*64
   j,m=self.changed(value,job)
   with self.subTest(name=name),self.assertRaises(ValueError):r.validate(j,m,self.authority)
 def test_outdated_restore_witness_rejected(self):
  value=copy.deepcopy(self.value);value['restore_witness']['version']=r.RESTORE_WITNESS
  j,m=self.changed(value)
  with self.assertRaises(ValueError):r.validate(j,m,self.authority)

 def test_actual_ten_scientific_files_required_no_invented_sampling_module(self):
  value=copy.deepcopy(self.value);original=copy.deepcopy(self.original);original['source_files'].pop('subnet/sampling_contract.py')
  value['original_signed_job']=self.sign(original);value['original_job_sha256']=r.sha(original);value['restore_witness']['science_source_files'].pop('subnet/sampling_contract.py');value['execution_runtime_source_files'].pop('subnet/sampling_contract.py')
  job=copy.deepcopy(self.job);job['source_files'].pop('subnet/sampling_contract.py');j,m=self.changed(value,job)
  r.validate(j,m,self.authority)
