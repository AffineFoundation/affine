import copy,importlib.util,json,time,types,unittest
from pathlib import Path
P=Path(__file__).parent
spec=importlib.util.spec_from_file_location('overlay_test',P/'mimo_diagnostic_quarantine.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
class QuarantineTests(unittest.TestCase):
 def setUp(self):
  self.now=3601;self.worker='b'*64;self.miner='c'*64;self.cp='d'*64;self.epoch='epoch-1';self.job={'job_id':'job-1'};self.key=q.digest(self.job);self.report={'epoch':self.epoch,'checkpoint':self.cp};self.request={'payload':self.report};self.row=dict(id='job-1',digest=self.key,worker=self.worker,report=self.report,report_digest=q.digest(self.report),report_request=self.request,envelope={'payload':self.job})
  self.admission=dict(verifier=self.worker,original_report_request_sha256=q.digest(self.request),observations=[dict(miner=self.miner,epoch=self.epoch,checkpoint=self.cp,outcome='confirmed_invalid')]);self.admitted={self.key:self.admission}
  self.p=dict(version=q.VERSION,authority='a'*64,created_at=3600,effective_cutoff=7200,expires_at=14400,entries=[dict(job_id='job-1',job_sha256=self.key,report_sha256=q.digest(self.report),report_request_sha256=q.digest(self.request),worker=self.worker,epoch=self.epoch,miner=self.miner,checkpoint=self.cp,diagnostic_sha256='e'*64,reason=q.REASON)])
 def apply(self,p=None,rows=None,admitted=None,cutoff=7200):return q.filter_admissions(self.admitted if admitted is None else admitted,[self.row]if rows is None else rows,self.p if p is None else p,cutoff=cutoff,clock=lambda:self.now)
 def test_exact_invalid_deferred_without_mutating_original(self):
  before=copy.deepcopy((self.admitted,self.row,self.p));kept,deferred=self.apply();self.assertEqual(kept,{});self.assertEqual(len(deferred),1);self.assertFalse(deferred[0]['validity_credit']);self.assertFalse(deferred[0]['fraud_claim']);self.assertFalse(deferred[0]['counts_as_numerical_coverage_failure']);self.assertEqual(before,(self.admitted,self.row,self.p))
 def test_before_effective_cutoff_unchanged(self):self.assertEqual(self.apply(cutoff=3600),(self.admitted,[]))
 def test_unlisted_admission_preserved(self):
  a=dict(self.admitted,other=copy.deepcopy(self.admission));kept,_=self.apply(admitted=a);self.assertEqual(kept,{'other':self.admission})
 def test_absent_or_unadmitted_report_not_promoted(self):self.assertEqual(self.apply(admitted={} ),({},[]))
 def test_expiry_fails_reader_instead_of_reenabling_penalty(self):
  self.now=14401
  with self.assertRaisesRegex(ValueError,'expired'):self.apply()
 def test_expired_install_still_reaches_writer_fallback(self):
  self.now=14401;b=types.SimpleNamespace(admit_queue_reports=lambda *a,**kw:(self.admitted,[]))
  def evidence(*a,**kw):b.admit_queue_reports([self.row],cutoff=7200);return dict(deferred=[],evidence_hashes={})
  b.evidence=evidence;q.install(b,self.p,authority=self.p['authority'],clock=lambda:self.now)
  with self.assertRaisesRegex(ValueError,'expired'):b.evidence()
 def test_wrong_report_rejected(self):
  r=copy.deepcopy(self.row);r['report']=dict(self.report,changed=True)
  with self.assertRaisesRegex(ValueError,'report/request'):self.apply(rows=[r])
 def test_wrong_worker_or_request_rejected(self):
  for field,value in [('worker','f'*64),('report_request',{'other':1})]:
   r=copy.deepcopy(self.row);r[field]=value
   with self.assertRaises(ValueError):self.apply(rows=[r])
 def test_valid_report_never_quarantined(self):
  a=copy.deepcopy(self.admitted);a[self.key]['observations'][0]['outcome']='verified_valid'
  with self.assertRaisesRegex(ValueError,'invalid group'):self.apply(admitted=a)
 def test_wrong_scope_rejected(self):
  for field,value in [('epoch','epoch-2'),('miner','f'*64),('checkpoint','f'*64)]:
   p=copy.deepcopy(self.p);p['entries'][0][field]=value
   with self.assertRaises(ValueError):self.apply(p=p)
 def test_duplicate_and_unbounded_policy_rejected(self):
  p=copy.deepcopy(self.p);p['entries']*=2
  with self.assertRaises(ValueError):self.apply(p=p)
  p=copy.deepcopy(self.p);p['expires_at']=1000000
  with self.assertRaises(ValueError):self.apply(p=p)
 def test_overlay_authenticates_first_then_preserves_counterfactual(self):
  calls=[];b=types.SimpleNamespace()
  def admit(rows,**kw):calls.append('original-authenticate');return dict(self.admitted),[{'other-refusal':True}]
  b.admit_queue_reports=admit
  def evidence(*args,**kw):
   admitted,refused=b.admit_queue_reports([self.row],cutoff=kw['cutoff']);return dict(snapshots=[{'legacy_factor':.9,'invalid_count':len(admitted)}],deferred=[],refused=refused,evidence_hashes={})
  b.evidence=evidence;q.install(b,self.p,authority=self.p['authority'],clock=lambda:self.now);r=b.evidence(cutoff=7200)
  self.assertEqual(calls,['original-authenticate']);self.assertEqual(r['snapshots'],[{'legacy_factor':.9,'invalid_count':0}]);self.assertEqual(r['refused'],[{'other-refusal':True}]);self.assertEqual(len(r['deferred']),1);self.assertIn('mimo_diagnostic_quarantine_sha256',r['evidence_hashes'])
 def test_original_auth_failure_propagates(self):
  b=types.SimpleNamespace()
  def admit(*a,**kw):raise ValueError('bad original signature')
  b.admit_queue_reports=admit;b.evidence=lambda **kw:b.admit_queue_reports([self.row],**kw)
  q.install(b,self.p,authority=self.p['authority'],clock=lambda:self.now)
  with self.assertRaisesRegex(ValueError,'bad original signature'):b.evidence(cutoff=7200)
if __name__=='__main__':unittest.main()
