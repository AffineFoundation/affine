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

class OriginalBinding(unittest.TestCase):
 def setUp(self):
  import base64
  from nacl.signing import SigningKey
  from subnet.storage import canonical
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.state=Path(self.tmp.name)
  self.key=SigningKey.generate();authority=self.key.verify_key.encode().hex()
  def sign(payload):return dict(payload=payload,signer=authority,signature=base64.b64encode(self.key.sign(canonical(payload)).signature).decode())
  self.manifest=dict(checkpoint={'id':'a'*64},source_bundle={'sha256':'b'*64},harness={'max_output_tokens':2048})
  self.request={'version':'sample-request'}
  job=dict(job_id='original',role='evaluate',manifest=sign(self.manifest),successor_calibration=self.request)
  self.envelope=sign(job);self.prior=dict(job_id='original',job_sha256=c.digest(job))
  for name,value in [('label.json',self.prior),('original-job.json',self.envelope),('original-report.json',{'success':True})]:
   (self.state/name).write_text(json.dumps(value))
  self.jobs=SimpleNamespace(state=self.state,controller=SimpleNamespace(authority=SimpleNamespace(id=authority)),checked=Mock(return_value={'success':True}),run=Mock(side_effect=AssertionError('must never dispatch')))
 def test_original_evidence_readback_only(self):
  repair.read_original(self.jobs,'label',self.manifest,self.request);self.jobs.checked.assert_called_once();self.jobs.run.assert_not_called()
 def test_changed_checkpoint_source_or_harness_rejected(self):
  for key,value in [('checkpoint',{'id':'c'*64}),('source_bundle',{'sha256':'d'*64}),('harness',{'max_output_tokens':1024})]:
   with self.assertRaisesRegex(ValueError,'original request binding'):
    repair.read_original(self.jobs,'label',dict(self.manifest,**{key:value}),self.request)
  self.jobs.checked.assert_not_called();self.jobs.run.assert_not_called()
 def test_changed_request_rejected(self):
  with self.assertRaisesRegex(ValueError,'original request binding'):repair.read_original(self.jobs,'label',self.manifest,{'version':'forged'})
 def test_tampered_original_signature_rejected(self):
  self.envelope['payload']['successor_calibration']={'version':'tampered'};(self.state/'original-job.json').write_text(json.dumps(self.envelope))
  with self.assertRaises(Exception):repair.read_original(self.jobs,'label',self.manifest,self.request)
  self.jobs.checked.assert_not_called()
