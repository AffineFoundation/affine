import copy,sys,unittest
from pathlib import Path
sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).resolve().parents[1])]
from test_unaudited_execution_contract import AmendmentTests
from subnet import unaudited_training_execution as a
from subnet.training_receipts import sha
class EffectiveLR(AmendmentTests):
 def test_exact_nested_grant_parent_job_and_release(self):
  value=self.run_check();grant=value['learning_rate_authorization']['payload']
  self.assertEqual(grant['job_id'],self.job['job_id']);self.assertEqual(grant['execution_release_sha256'],value['execution_release_sha256']);self.assertEqual(grant['effective_learning_rate'],5e-7)
 def test_resigned_other_rate_inside_grant_refused(self):
  grant=copy.deepcopy(self.value['learning_rate_authorization']['payload']);grant['effective_learning_rate']=1e-6;self.value['learning_rate_authorization']=self.sign(grant)
  with self.assertRaisesRegex(ValueError,'exact LR grant'):self.run_check()
 def test_lr_changed_without_qualification_rejected(self):
  self.value['effective_learning_rate']=1e-6
  with self.assertRaisesRegex(ValueError,'qualification'):self.run_check()
 def test_unchanged_high_lr_not_a_corrective_release(self):
  self.value['effective_learning_rate']=1e-5
  with self.assertRaisesRegex(ValueError,'corrective'):self.run_check()
 def test_cross_job_or_parent_grant_cannot_reuse(self):
  for key,value in [('job_id','another-job'),('parent_descriptor_sha256','7'*64),('execution_release_sha256','7'*64),('optimizer_step_before',1)]:
   original=copy.deepcopy(self.value['learning_rate_authorization']);grant=copy.deepcopy(original['payload']);grant[key]=value;self.value['learning_rate_authorization']=self.sign(grant)
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,'exact LR grant'):self.run_check()
   self.value['learning_rate_authorization']=original
 def test_report_truthfully_records_lr_and_actual_state_version(self):
  report=a.provenance(self.envelope(),self.authority)
  self.assertEqual(report['effective_learning_rate'],5e-7);self.assertEqual(report['learning_rate_authorization_sha256'],sha(self.value['learning_rate_authorization']));self.assertEqual(report['state_version'],'persistent-fp32-trainer-state-v2-effective-lr')
 def test_expired_historical_report_remains_authenticated(self):
  report=a.provenance(self.envelope(),self.authority);self.assertEqual(report['effective_learning_rate'],5e-7)
  with self.assertRaisesRegex(ValueError,'valid|lifetime'):a.validate(self.envelope(),self.authority,now=121)
if __name__=='__main__':unittest.main()
