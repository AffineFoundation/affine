import copy
import unittest
from test_numerical_resolution import ReviewedUnknownControls
from subnet.numerical_resolution import apply

class DeferredOriginals(ReviewedUnknownControls):
    def test_missing_job_is_explicitly_deferred_not_valid_or_fraud(self):
        deferred=[]
        result=apply([], [self.row], {}, **self.kw, unavailable_execution_deferrals=deferred)
        self.assertEqual(result,[])
        self.assertEqual(len(deferred),1)
        self.assertEqual(deferred[0]['status'],'unresolved')
        self.assertEqual(deferred[0]['original_job_sha256'],self.entry['original_job_sha256'])
    def test_missing_job_does_not_suppress_unrelated_confirmed_invalid(self):
        other=dict(self.observation,evidence_id='d'*64,job_sha256='e'*64)
        result=apply([other],[self.row],{},**self.kw,unavailable_execution_deferrals=[])
        self.assertEqual(result,[other])
        self.assertEqual(result[0]['outcome'],'confirmed_invalid')
    def test_present_observation_without_admission_still_rejected(self):
        with self.assertRaisesRegex(ValueError,'observation lacks original admission'):
            apply([self.observation],[self.row],{},**self.kw,unavailable_execution_deferrals=[])
    def test_snapshot_keeps_missing_job_unresolved_and_collects_deferral(self):
        from subnet.continuous_audit_policy import snapshot,digest
        deferred=[]
        snap=snapshot([self.row],[],{self.verifier:['verify']},epoch='e29',round=29,
            checkpoint='1'*64,cutoff=3600,audit_policy=self.audit_policy,admitted_jobs={},
            authority=self.authority,numerical_resolution_policy=self.document,
            expected_numerical_resolution_policy_sha256=digest(self.document),
            numerical_reference_archives=self.archives,
            numerical_unavailable_execution_deferrals=deferred)
        miner=snap['miners'][self.row['miner']]
        self.assertEqual(miner['confirmed_invalid_current'],0)
        self.assertEqual(miner['resolved_current'],0)
        self.assertEqual(miner['validity_probability'],.5)
        self.assertFalse(snap['unaudited_samples_claimed_verified'])
        self.assertEqual(len(deferred),1)
    def test_missing_job_remains_strict_without_explicit_deferral_sink(self):
        with self.assertRaisesRegex(ValueError,'original admitted execution'):
            apply([],[self.row],{},**self.kw)
    def test_bad_signature_still_rejected_when_deferral_enabled(self):
        kw=copy.deepcopy(self.kw);kw['policy_document']['signature']='A'*88
        with self.assertRaises(Exception):
            apply([],[self.row],{},**kw,unavailable_execution_deferrals=[])

if __name__=='__main__':unittest.main()
