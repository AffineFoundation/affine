import json
import tempfile
import unittest
import types
from unittest import mock
from pathlib import Path

from subnet.eval_suite import checkpoints, run_id, retry_due, evaluate_available, canonical
import hashlib


class SuiteEvidenceTests(unittest.TestCase):
    def test_transient_failure_recovers_and_keeps_both_attempts(self):
        model = types.ModuleType('subnet.model')
        files = {'model.safetensors':'approved-bytes'}
        commitment = hashlib.sha256(canonical(files)).hexdigest()
        model.model_files = lambda _: files
        model.check_runtime_profile = lambda _: None
        model.NUMERICAL_RUNTIME_REVISION = 'test-pinned-runtime'
        evaluation = types.ModuleType('subnet.evaluation')
        def outcome(manifest, path, definition, heldout, destination):
            return dict(run_id=heldout['run_id'],status='complete',timestamp=1300,count=1,mean_reward=0)
        evaluation.evaluate = mock.Mock(side_effect=[TimeoutError('temporary transport failure'),None])
        with tempfile.TemporaryDirectory() as temporary:
            config = dict(output=temporary,checkpoint_state='unused',runtime_profile={},suites=[dict(
                spec={'id':'math','version':'v1'},harness={'version':'plain-transcript-v1','policy':'autoregressive'},
                training_indices=[0],heldout_indices=[1])])
            approved = [dict(checkpoint=commitment,path='/checkpoint',epoch_id='nonpayable-test',training_steps=0)]
            with mock.patch.dict('sys.modules', {'subnet.model':model,'subnet.evaluation':evaluation}), mock.patch('subnet.eval_suite.checkpoints',return_value=approved):
                with mock.patch('subnet.eval_suite.time.time',return_value=1000):
                    first = evaluate_available(config)
                evaluation.evaluate.side_effect = outcome
                with mock.patch('subnet.eval_suite.time.time',return_value=1300):
                    second = evaluate_available(config)
                    third = evaluate_available(config)
            self.assertEqual(first[0]['status'],'error')
            self.assertEqual(second[0]['status'],'complete')
            self.assertEqual(third,[])
            attempts = sorted((Path(temporary)/'attempts').glob('*.json'))
            self.assertEqual([json.loads(p.read_text())['status'] for p in attempts],['error','complete'])
            self.assertEqual(evaluation.evaluate.call_count,2)

    def test_wrong_checkpoint_cannot_be_reused_by_later_environments(self):
        model = types.ModuleType('subnet.model')
        model.model_files = mock.Mock(return_value={'model.safetensors':'altered-bytes'})
        model.check_runtime_profile = lambda _: None
        model.NUMERICAL_RUNTIME_REVISION = 'test-pinned-runtime'
        evaluation = types.ModuleType('subnet.evaluation')
        evaluation.evaluate = mock.Mock(side_effect=AssertionError('unapproved weights evaluated'))
        with tempfile.TemporaryDirectory() as temporary:
            config = dict(output=temporary, checkpoint_state='unused', runtime_profile={}, suites=[
                dict(spec={'id':env,'version':'v1'}, harness={'version':'plain-transcript-v1','policy':'autoregressive'},
                     training_indices=[0],heldout_indices=[1]) for env in ('first','second')])
            approved = [dict(checkpoint='approved-commitment',path='/checkpoint',epoch_id='nonpayable-test',training_steps=0)]
            with mock.patch.dict('sys.modules', {'subnet.model':model,'subnet.evaluation':evaluation}), mock.patch('subnet.eval_suite.checkpoints',return_value=approved):
                completed = evaluate_available(config)
            self.assertEqual([r['status'] for r in completed], ['error','error'])
            evaluation.evaluate.assert_not_called()
            self.assertEqual(model.model_files.call_count,2)

    def test_failed_evaluations_recover_without_repeating_completed_evidence(self):
        record = {'status': 'error', 'timestamp': 100, 'evaluation_attempt': 1}
        self.assertFalse(retry_due(record, 399, 300, 3))
        self.assertTrue(retry_due(record, 400, 300, 3))
        self.assertFalse(retry_due(dict(record, evaluation_attempt=3), 1000, 300, 3))
        self.assertFalse(retry_due(dict(record, status='complete'), 1000, 300, 3))
        for malformed in (dict(record, timestamp=float('nan')), dict(record, evaluation_attempt=True), {}):
            self.assertFalse(retry_due(malformed, 1000, 300, 3))

    def test_completed_checkpoint_selection_skips_failed_epochs(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            (state/'epoch-stage-0.json').write_text(json.dumps(dict(checkpoint={'id':'base'},
                checkpoint_path='/approved/base',manifest={'epoch':'nonpayable-first'})))
            (state/'progress.json').write_text(json.dumps({'history':[
                {'epoch_id':'no-data','accepted_batches':0},
                {'epoch_id':'trained','accepted_batches':1,'training':{'checkpoint':'new','steps':2,'weights_changed':True}},
                {'epoch_id':'unchanged','accepted_batches':1,'training':{'checkpoint':'bad','steps':1,'weights_changed':False}},
            ]}))
            rows = checkpoints(state)
            self.assertEqual([r['checkpoint'] for r in rows], ['base','new'])
            self.assertEqual(rows[-1]['path'],str(state/'checkpoint-2'))
            self.assertEqual(rows[-1]['training_steps'],2)

    def test_record_identity_pins_policy_tasks_profile_and_checkpoint(self):
        definition={'env_id':'math','harness':{'policy':'autoregressive'}}
        heldout={'indices':[2,3],'seed':42,'repeats':1}
        profile={'OMP_NUM_THREADS':'4'}
        baseline=run_id('checkpoint-a',definition,heldout,profile)
        self.assertEqual(baseline,run_id('checkpoint-a',definition,heldout,profile))
        for value in [run_id('checkpoint-b',definition,heldout,profile),
                      run_id('checkpoint-a',definition,dict(heldout,indices=[4,5]),profile),
                      run_id('checkpoint-a',definition,heldout,{'OMP_NUM_THREADS':'2'}),
                      run_id('checkpoint-a',definition,heldout,profile,'bounded-toploc-v2'),
                      run_id('checkpoint-a',dict(definition,harness={'policy':'candidates'}),heldout,profile)]:
            self.assertNotEqual(baseline,value)


if __name__ == '__main__':
    unittest.main()
