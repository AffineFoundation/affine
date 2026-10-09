import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from test_successor_calibration import Controls
from ops import completed_calibration_retry as repair
from subnet import successor_calibration as c
from subnet.forced_sampling import new_contract
from subnet.fast_prefill_audit import VERSION

class Readback(Controls):
 def setUp(self):
  super().setUp();self.key='a'*64
  self.policy=c.admitted_policy(self.result,self.m,self.req)
  self.config=dict(sampling_policy=dict(version=VERSION,max_attempts=16,calibration=self.policy))
  self.finalreq=dict(self.req,draw_contract=new_contract(self.config['sampling_policy']))
  self.seed=dict(job_id='seed',completed_at=90,success=True,successor_calibration=self.result)
  self.confirm=dict(job_id='confirm',completed_at=110,success=True,successor_calibration=dict(self.result,request_sha256=c.digest(self.finalreq)))
  self.row=dict(label='successor-reconfirm-'+self.key[:24]+'-0',request=self.finalreq,policy=self.policy,report=self.confirm)
  self.record=dict(original_job_id='seed',report_sha256=c.digest(self.seed),confirmation_original_job_id='confirm',confirmation_sha256=c.digest(self.confirm),calibration=self.policy,bounded_confirmations=dict(version=c.RECALIBRATION_VERSION,created_at=100,deadline=200,rounds=[self.row]))
  self.old=Mock(side_effect=ValueError('original strict deadline'))
 def invoke(self):
  def read(jobs,label,manifest,request):
   return self.seed if label.startswith('successor-calibration-')else self.confirm
  with patch.object(repair,'read_original',side_effect=read):
   return repair.completed(self.old,None,self.config,{},self.record,None,self.key,self.m,self.req,self.seed,None,None,dict(max_confirmations=4,deadline_seconds=100))
 def test_completed_readback_after_deadline_unchanged(self):
  before=copy.deepcopy(self.record)
  with patch('time.time',return_value=100000):result=self.invoke()
  self.assertEqual(result['sampling_policy']['calibration'],self.policy);self.assertEqual(self.record,before);self.old.assert_not_called()
 def test_late_or_exact_deadline_rejected(self):
  for value in (200,201):
   self.confirm['completed_at']=value;self.record['confirmation_sha256']=c.digest(self.confirm)
   with self.assertRaisesRegex(ValueError,'late'):self.invoke()
 def test_prejournal_confirmation_rejected(self):
  self.confirm['completed_at']=99;self.record['confirmation_sha256']=c.digest(self.confirm)
  with self.assertRaisesRegex(ValueError,'late'):self.invoke()
 def test_incomplete_stays_original_strict_path(self):
  del self.record['confirmation_sha256']
  with self.assertRaisesRegex(ValueError,'strict deadline'):self.invoke()
  self.old.assert_called_once()
 def test_seed_record_tamper_rejected(self):
  self.record['report_sha256']='f'*64
  with self.assertRaisesRegex(ValueError,'seed changed'):self.invoke()
 def test_bound_not_widened(self):
  self.row['policy']=dict(self.policy,cdf_abs_error=.1)
  with self.assertRaisesRegex(ValueError,'proposal changed'):self.invoke()
 def test_missing_report_never_dispatches(self):
  with patch.object(repair,'read_original',side_effect=FileNotFoundError):
   with self.assertRaises(FileNotFoundError):repair.completed(self.old,None,self.config,{},self.record,None,self.key,self.m,self.req,self.seed,None,None,dict(max_confirmations=4,deadline_seconds=100))
  self.old.assert_not_called()
 def test_checked_rejection_not_suppressed(self):
  with patch.object(repair,'read_original',side_effect=ValueError('remote role report binding')):
   with self.assertRaisesRegex(ValueError,'report binding'):repair.completed(self.old,None,self.config,{},self.record,None,self.key,self.m,self.req,self.seed,None,None,dict(max_confirmations=4,deadline_seconds=100))
 def test_failed_confirmation_cannot_reuse(self):
  self.confirm['successor_calibration']['reports']=copy.deepcopy(self.reports)
  self.confirm['successor_calibration']['reports'][0]['measured_cdf_abs_error']=2e-5
  self.record['confirmation_sha256']=c.digest(self.confirm)
  with self.assertRaisesRegex(ValueError,'never confirmed'):self.invoke()
