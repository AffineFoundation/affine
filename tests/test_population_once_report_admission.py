import copy,unittest
from unittest.mock import patch
import test_continuous_audit_policy as fixture
from subnet import continuous_audit_policy as policy
from subnet.continuous_audit_service import admit_completed_reports,BACKEND_DEFERRAL_POLICY
class PopulationOnceControls(unittest.TestCase):
 def setUp(self):self.f=fixture.PolicyControls();self.f.setUp();self.q,self.pins=self.f.queue_fixture()
 def call(self,rows,records=None,defer=None):return admit_completed_reports(rows,records if records is not None else[self.f.row],self.f.root,{self.f.worker:['verify']},self.pins,execution_evidence_policy=None,cutoff=30,deferral_policy=defer)
 def test_many_originals_validate_population_exactly_once(self):
  with patch.object(policy,'population',wraps=policy.population)as validate:
   admissions,deferred=self.call([self.q]*8);self.assertEqual(validate.call_count,1);self.assertEqual(len(admissions),1);self.assertEqual(deferred,[])
 def test_duplicated_or_corrupt_population_is_never_bypassed(self):
  for rows in [[self.f.row,self.f.row],[dict(self.f.row,checkpoint='invalid')]]:
   with self.assertRaises(ValueError):self.call([self.q],rows)
 def test_bad_signature_or_lease_not_masked_as_neutral_deferral(self):
  for field,value in [('token','wrong'),('report_request',fixture.signed(__import__('nacl').signing.SigningKey.generate(),self.q['report_request']['payload']))]:
   q=copy.deepcopy(self.q);q[field]=value
   with self.assertRaises(Exception):self.call([q],defer=BACKEND_DEFERRAL_POLICY)
 def test_only_typed_backend_gate_can_be_deferred(self):
  q,pins,ep=self.f.standard_fixture();self.pins=pins
  self.assertFalse(q['report']['execution_resources_enforced'])
  with self.assertRaises(policy.BackendEvidenceNotAdmitted):self.call([q])
  admitted,deferred=self.call([q],defer=BACKEND_DEFERRAL_POLICY);self.assertEqual(admitted,{});self.assertEqual(len(deferred),1);self.assertFalse(deferred[0]['validity_credit']);self.assertFalse(deferred[0]['fraud_claim'])
 def test_order_and_repeated_original_digests_equivalent(self):
  one=self.call([self.q]);batch=self.call([self.q,self.q]);self.assertEqual(one,batch)
