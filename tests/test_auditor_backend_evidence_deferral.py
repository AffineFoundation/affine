import copy,unittest
from test_continuous_audit_policy import PolicyControls,signed
from subnet.continuous_audit_policy import admit_queue_reports,BackendEvidenceNotAdmitted,digest
from subnet.continuous_audit_service import admit_completed_reports,BACKEND_DEFERRAL_POLICY,prevalidate_source_execution_rows,SOURCE_EXECUTION_PREFLIGHT
class DeferralControls(unittest.TestCase):
 setUp=PolicyControls.setUp
 queue_fixture=PolicyControls.queue_fixture
 standard_fixture=PolicyControls.standard_fixture
 def test_default_refusal_and_opted_in_no_credit(self):
  q,pins,ep=self.standard_fixture()
  with self.assertRaises(BackendEvidenceNotAdmitted):admit_completed_reports([q],[self.row],self.root,{self.worker:['verify']},pins,execution_evidence_policy=None,cutoff=30)
  a,d=admit_completed_reports([q],[self.row],self.root,{self.worker:['verify']},pins,execution_evidence_policy=None,cutoff=30,deferral_policy=BACKEND_DEFERRAL_POLICY)
  self.assertEqual(a,{});self.assertEqual(len(d),1);self.assertFalse(d[0]['validity_credit']);self.assertFalse(d[0]['fraud_claim'])
 def test_exact_v2_source_cutoff_and_old_rows_unchanged(self):
  q,pins,ep=self.standard_fixture();ep['version']='explicit-backend-execution-evidence-v2';ep['sources']['9'*64]['effective_cutoff']=40
  a,d=admit_completed_reports([q],[self.row],self.root,{self.worker:['verify']},pins,execution_evidence_policy=ep,cutoff=30,deferral_policy=BACKEND_DEFERRAL_POLICY);self.assertFalse(a);self.assertEqual(len(d),1)
  a,d=admit_completed_reports([q],[self.row],self.root,{self.worker:['verify']},pins,execution_evidence_policy=ep,cutoff=40,deferral_policy=BACKEND_DEFERRAL_POLICY);self.assertTrue(a);self.assertFalse(d)
 def test_source_runtime_profile_signature_mismatch_never_neutralized(self):
  q,pins,ep=self.standard_fixture()
  for key,value in [('source_files',{'forged':'0'*64}),('runtime_versions',{'torch':'forged'}),('backend_profile',{'forged':True}),('numerical_policy','forged')]:
   bad=copy.deepcopy(q);bad['report'][key]=value;bad['report_digest']=digest(bad['report']);bad['report_request']=signed(self.key,dict(action='report',job_id='job-1',token='lease-token',report=bad['report']))
   with self.assertRaises(ValueError):admit_completed_reports([bad],[self.row],self.root,{self.worker:['verify']},pins,execution_evidence_policy=None,cutoff=30,deferral_policy=BACKEND_DEFERRAL_POLICY)
  bad=copy.deepcopy(q);bad['report_request']['signature']='not-a-signature'
  with self.assertRaises(Exception):admit_completed_reports([bad],[self.row],self.root,{self.worker:['verify']},pins,execution_evidence_policy=None,cutoff=30,deferral_policy=BACKEND_DEFERRAL_POLICY)
 def test_complete_source_preflight_catches_additive_omission(self):
  q,pins,ep=self.standard_fixture();job=q['envelope']['payload'];s=dict(source_execution_evidence_preflight=SOURCE_EXECUTION_PREFLIGHT,execution_evidence_policy=ep,approved_sources=pins,job_metadata={'9'*64:dict(source_files=job['source_files'],runtime_versions=job['runtime_versions'])});prevalidate_source_execution_rows(s)
  for mutation in ['missing','hash','runtime']:
   x=copy.deepcopy(s)
   if mutation=='missing':x['approved_sources']['a'*64]=pins['9'*64];x['job_metadata']['a'*64]=x['job_metadata']['9'*64]
   elif mutation=='hash':x['execution_evidence_policy']['sources']['9'*64]['backend_module_sha256']='0'*64
   else:x['execution_evidence_policy']['sources']['9'*64]['runtime_versions']={'torch':'wrong'}
   with self.assertRaises(ValueError):prevalidate_source_execution_rows(x)
if __name__=='__main__':unittest.main()
