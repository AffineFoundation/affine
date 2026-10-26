"""Synthetic read-only sampling observer controls; no GPU, keys or requests."""
import copy
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HELPER = Path(__file__).resolve().parents[1] / 'state/root-audits/run_e9_sampling_diagnostic_20261004_v1.py'
MODULE = runpy.run_path(str(HELPER), run_name='sampling_diagnostic_test_definitions') if HELPER.is_file() else None


@unittest.skipIf(MODULE is None, 'operator-local sampling diagnostic helper is absent')
class E9SamplingDiagnosticReviewTests(unittest.TestCase):
    def test_bounded_hashes_lengths_and_first_difference_preserve_inputs(self):
        expected=[1,2,3]
        for submitted, first, kind in (([1,2,3],None,'none'),([1,8,3],1,'different-token'),
                                       ([1,2],2,'length-prefix'),([1,2,3,4],3,'length-prefix')):
            originals=copy.deepcopy((submitted,expected))
            result=MODULE['sampling_statistics'](submitted,expected)
            self.assertEqual(result['exact_tokens_match'],submitted==expected)
            self.assertEqual(result['first_differing_token_index'],first)
            self.assertEqual(result['mismatch_kind'],kind)
            self.assertEqual(result['submitted_tokens_sha256'],MODULE['sha'](submitted))
            self.assertEqual(result['replayed_tokens_sha256'],MODULE['sha'](expected))
            self.assertFalse(any(isinstance(value,list) for value in result.values()))
            self.assertEqual((submitted,expected),originals)
        with self.assertRaisesRegex(ValueError,'bounded'):
            MODULE['sampling_statistics']([1]*8193,expected)

    def runtime(self,returned,records):
        class SyntheticRuntime:
            runtime_revision='synthetic-exact-revision'
            sampling_context={'binding':'unchanged'}
            spec=SimpleNamespace(id='synthetic-environment')
            def sample_output(self,prompt,seed,messages,turn,index,task_hash):
                return returned
            def verify(self,output):
                return self.sample_output([10,20],0,[],0,3,'f'*64)
        SyntheticRuntime.sample_output=MODULE['observe_sample_output'](
            SyntheticRuntime.sample_output,SyntheticRuntime.verify.__code__,records)
        return SyntheticRuntime()

    def test_observer_returns_same_token_object_and_records_exact_original_replay(self):
        returned=[1,2,3]; records=[]
        actual=self.runtime(returned,records).verify([1,8,3])
        self.assertIs(actual,returned)
        self.assertEqual(returned,[1,2,3])
        self.assertTrue(records[0]['instrumentation_complete'])
        self.assertEqual(records[0]['first_differing_token_index'],1)
        self.assertFalse(records[0]['exact_tokens_match'])
        self.assertEqual(records[0]['runtime_revision'],'synthetic-exact-revision')
        self.assertEqual(records[0]['sampling_context_sha256'],MODULE['sha']({'binding':'unchanged'}))

    def test_instrumentation_fault_and_original_exception_do_not_change_return_or_error(self):
        returned=[1,2,3]; records=[]
        self.assertIs(self.runtime(returned,records).verify(object()),returned)
        self.assertFalse(records[0]['instrumentation_complete'])
        failure=RuntimeError('synthetic original sampling failure')
        def original(*args):raise failure
        wrapped=MODULE['observe_sample_output'](original,None,records)
        with self.assertRaises(RuntimeError)as observed:
            wrapped(object(),[],0,[],0,0,'f'*64)
        self.assertIs(observed.exception,failure)

    def test_only_identity_time_and_exact_get_refresh_allowed(self):
        original=dict(job_id='original',role='verify',created_at=0,expires_at=1,
            manifest={'payload':{'epoch':'original-epoch','checkpoint':{'id':'a'*64,'files':{'a':'b'},'read_urls':{'a':'old'}},'sampling_contract':{'seed_binding':'original'}}},
            submissions=[{'sha256':'f'*64,'url':'old'}],runtime_versions={'torch':'same'})
        diagnostic=copy.deepcopy(original)
        diagnostic.update(job_id=MODULE['DIAGNOSTIC_JOB'],created_at=2,expires_at=3)
        diagnostic['manifest']['payload']['checkpoint']['read_urls']={'a':'new'}
        diagnostic['submissions'][0]['url']='new'
        with patch.dict(MODULE['validate_derivation'].__globals__,authenticate=lambda envelope,authority:envelope['payload']):
            MODULE['validate_derivation'](diagnostic,original,'synthetic')
            for key,value in (('sampling_contract',{'seed_binding':'different'}),
                              ('checkpoint',{'id':'c'*64,'files':{'a':'b'},'read_urls':{'a':'new'}})):
                changed=copy.deepcopy(diagnostic);changed['manifest']['payload'][key]=value
                with self.assertRaisesRegex(ValueError,'scientific verify request changed'):
                    MODULE['validate_derivation'](changed,original,'synthetic')


if __name__=='__main__':unittest.main()
