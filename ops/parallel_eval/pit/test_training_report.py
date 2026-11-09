#!/usr/bin/env python3
"""Training-report copy-back + GPU helpers."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from .box import BoxRunner, ScriptedBoxIO
from .binding import dequeue_and_bind
from .config import PitConfig
from .pe_lane import LaneService
from .state import ItemRecord, ItemState, apply_transition
from .training_report import (
    TrainingReportError,
    copy_training_report_to_host,
    detect_local_gpu,
    extract_report_summary,
    gpu_type_of,
)


H0, M0 = 'a' * 64, 'b' * 64


class TestTrainingReport(unittest.TestCase):
    def test_extract_summary_keys(self):
        raw = {
            'file_summary': {
                'source_file_hashes': {'batch': 'b' * 64},
                'tasks': 125,
                'rollouts': 1000,
                'prompt_tokens': 10,
                'output_tokens': 20,
                'loss_tokens': 15,
            },
            'optimizer': {'sgd': {'raw_update_norm': 0.42}},
            'preclip_gradient_norm': 1.5,
            'engine': 'packed',
        }
        summary = extract_report_summary(raw)
        self.assertEqual(summary['update_size'], 0.42)
        self.assertEqual(summary['gradient_norm'], 1.5)
        self.assertEqual(summary['batching']['tasks'], 125)
        self.assertEqual(summary['tokens']['prompt_tokens'], 10)

    def test_copy_to_host_with_pe_state(self):
        with tempfile.TemporaryDirectory() as td:
            report = Path(td) / 'report.json'
            report.write_text(json.dumps({
                'source_file_hashes': {'batch': 'b' * 64},
                'batching': {'tasks': 1, 'rollouts': 8},
                'tokens': {'loss_tokens': 3},
                'update_size': 0.1,
                'gradient_norm': 0.2,
            }))
            with mock.patch.dict(os.environ, {'PE_STATE': td}):
                summary = copy_training_report_to_host(
                    report_path=str(report),
                    item_id='x' * 32,
                    epoch=7,
                    candidate_id='c' * 64,
                )
            self.assertTrue(summary.get('host_path'))
            stored = Path(summary['host_path'])
            self.assertTrue(stored.is_file())
            payload = json.loads(stored.read_text())
            self.assertEqual(payload['epoch'], 7)
            self.assertEqual(payload['summary']['update_size'], 0.1)

    def test_missing_report_fail_closed(self):
        with self.assertRaises(TrainingReportError):
            copy_training_report_to_host(
                report_path='/no/such/report.json',
                item_id='x' * 32,
                epoch=1,
            )

    def test_gpu_type_of(self):
        self.assertEqual(gpu_type_of('NVIDIA B300'), 'B300')
        self.assertEqual(gpu_type_of({'gpu': 'H200-SXM'}), 'H200')
        self.assertIsNone(gpu_type_of(None))

    def test_detect_local_gpu_handles_missing_smi(self):
        with mock.patch('ops.parallel_eval.pit.training_report.subprocess.run', side_effect=OSError('no smi')):
            self.assertIsNone(detect_local_gpu())

    def test_box_runner_stores_report_and_gpus(self):
        r = ItemRecord(item_id='x' * 32, registration_index=1, epoch=7, hotkey_prefix='5A')
        r = dequeue_and_bind(
            r, head=H0, head_epoch=1, round_id='r', round_seed_commitment='c',
            master_digest=M0, dequeued_at_mono=1.0,
        )
        r = apply_transition(r, ItemState.RENTING)
        r = apply_transition(
            r, ItemState.PROVISIONING,
            updates={'box_id': 'box-1', 'meta': {'train_gpu': 'B300', 'eval_gpu': 'B300'}},
        )
        # ItemRecord.meta may be set via apply_transition updates differently —
        # stamp meta directly for the GPU path.
        r.meta = dict(r.meta or {}, train_gpu='B300', eval_gpu='B300')
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(os.environ, {'PE_STATE': td}):
                runner = BoxRunner(
                    config=PitConfig(enabled=True, mode='shadow'),
                    io=ScriptedBoxIO(),
                    lane=LaneService(),
                )
                rec, bundle = runner.run(
                    r, batch_handle='batch-handle:x', plan_handle='plan-handle',
                    train_gpu='B300', eval_gpu='B300',
                )
        self.assertEqual(rec.state, ItemState.RESULT)
        self.assertEqual(bundle.train_gpu, 'B300')
        self.assertEqual(bundle.eval_gpu, 'B300')
        self.assertEqual(bundle.probe_result.get('train_gpu'), 'B300')
        self.assertEqual(bundle.probe_result.get('eval_gpu'), 'B300')
        self.assertIn('source_file_hashes', bundle.training_report)
        self.assertEqual(bundle.training_report.get('update_size'), 1.0)


if __name__ == '__main__':
    unittest.main()
