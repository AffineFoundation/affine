"""Synthetic aggregate/observer controls; no source execution, network or GPU."""
import copy
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch
import numpy as np


HELPER = Path(__file__).resolve().parents[1]/'state/root-audits/run_e9_probability_diagnostic_20261004_v1.py'
MODULE = runpy.run_path(str(HELPER), run_name='diagnostic_test_definition') if HELPER.is_file() else None


@unittest.skipIf(MODULE is None, 'operator-local diagnostic helper is absent')
class E9ProbabilityDiagnosticReviewTests(unittest.TestCase):
    def test_stats_match_exact_original_predicate_without_mutating_arrays(self):
        summarize = MODULE['probability_statistics']
        expected = np.array([[-1.,-2.,-3.],[-4.,-5.,-6.]], dtype=np.float32)
        for label, claimed in (
                ('same',expected.copy()),
                ('below',expected + np.float32(5e-6)),
                ('above',expected + np.float32(2e-5)),
                ('nonfinite',np.array([[np.nan,np.inf,-np.inf],[-4.,-5.,-6.]],dtype=np.float32)),
                ('shape',expected[:1].copy())):
            with self.subTest(label=label):
                before = claimed.copy(); expected_before = expected.copy()
                result = summarize(claimed, expected)
                predicate = claimed.shape == expected.shape and np.isfinite(claimed).all() and np.allclose(claimed,expected,atol=1e-5,rtol=0)
                self.assertEqual(result['original_probability_predicate_passes'], bool(predicate))
                np.testing.assert_array_equal(before, claimed)
                np.testing.assert_array_equal(expected_before, expected)
                if claimed.shape == expected.shape:
                    self.assertEqual(result['out_of_tolerance_count'],int(np.count_nonzero(~np.isclose(claimed,expected,atol=1e-5,rtol=0))))
                self.assertNotIn('claimed_array',result)

    def test_compute_observer_returns_same_original_objects(self):
        acts = [object()]; probabilities = np.array([[-1.,-2.]], dtype=np.float32)
        records=[]
        class SyntheticRuntime:
            runtime_revision='synthetic-exact-revision'
            def compute(self,prompt,output):
                return acts,probabilities
            def verify(self,claimed):
                i=0
                return self.compute([1,2],[3])
        original = SyntheticRuntime.compute
        SyntheticRuntime.compute = MODULE['observe_compute'](original,SyntheticRuntime.verify.__code__,records)
        returned = SyntheticRuntime().verify(probabilities.copy())
        self.assertIs(returned[0],acts);self.assertIs(returned[1],probabilities)
        self.assertTrue(records[0]['instrumentation_complete'])
        self.assertTrue(records[0]['original_probability_predicate_passes'])
        self.assertEqual(records[0]['runtime_revision'],'synthetic-exact-revision')

    def test_instrumentation_error_does_not_change_compute_return(self):
        acts = [object()]; probabilities = np.array([[-1.,-2.]], dtype=np.float32); records=[]
        class SyntheticRuntime:
            runtime_revision='synthetic-exact-revision'
            def compute(self,prompt,output):return acts,probabilities
            def verify(self,claimed):return self.compute([1],[2])
        original = SyntheticRuntime.compute
        SyntheticRuntime.compute = MODULE['observe_compute'](original,SyntheticRuntime.verify.__code__,records)
        # A malformed synthetic object forces the aggregate helper to fail.
        returned = SyntheticRuntime().verify(object())
        self.assertIs(returned[0],acts);self.assertIs(returned[1],probabilities)
        self.assertFalse(records[0]['instrumentation_complete'])

    def test_diagnostic_derivation_allows_only_identity_time_and_get_refresh(self):
        namespace = MODULE['validate_derivation'].__globals__
        original=dict(job_id='original',role='verify',created_at=0,expires_at=1,
            manifest={'payload':{'epoch':'original-epoch','checkpoint':{'id':'a'*64,'files':{'a':'b'},'read_urls':{'a':'old'}},'sampling_contract':{'randomness':'original'}}},
            submissions=[{'sha256':'f'*64,'url':'old'}],runtime_versions={'torch':'same'})
        diagnostic=copy.deepcopy(original)
        diagnostic.update(job_id=MODULE['DIAGNOSTIC_JOB'],created_at=2,expires_at=3)
        diagnostic['manifest']['payload']['checkpoint']['read_urls']={'a':'new'}
        diagnostic['submissions'][0]['url']='new'
        # Authentication is independently checked by the real root-signed entry
        # path. This isolated unit covers the allowed scientific delta.
        with patch.dict(namespace, authenticate=lambda envelope,authority:envelope['payload']):
            MODULE['validate_derivation'](diagnostic,original,'synthetic')
            diagnostic['manifest']['payload']['sampling_contract']['randomness']='changed'
            with self.assertRaisesRegex(ValueError,'scientific verify request changed'):
                MODULE['validate_derivation'](diagnostic,original,'synthetic')


if __name__ == '__main__':unittest.main()
