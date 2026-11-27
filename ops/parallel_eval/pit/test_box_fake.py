#!/usr/bin/env python3
import unittest

from .binding import dequeue_and_bind
from .box import BoxRunner, Fp32MasterError, ScriptedBoxIO
from .config import PitConfig
from .pe_lane import LaneService
from .state import ItemRecord, ItemState, apply_transition
from .train_box import TrainBoxError, assert_no_bootstrap_override, run_train_box


H0, M0, C1 = 'a' * 64, 'b' * 64, 'c' * 64


class TestBoxFake(unittest.TestCase):
    def _item(self):
        r = ItemRecord(item_id='x' * 32, registration_index=1, epoch=7, hotkey_prefix='5A')
        r = dequeue_and_bind(
            r, head=H0, head_epoch=1, round_id='r', round_seed_commitment='c',
            master_digest=M0, dequeued_at_mono=1.0,
        )
        r = apply_transition(r, ItemState.RENTING)
        r = apply_transition(r, ItemState.PROVISIONING, updates={'box_id': 'box-1'})
        return r

    def test_happy_path_to_result(self):
        runner = BoxRunner(config=PitConfig(enabled=True, mode='shadow'), io=ScriptedBoxIO(), lane=LaneService())
        rec, bundle = runner.run(
            self._item(), batch_handle='batch-handle:x', plan_handle='plan-handle',
        )
        self.assertEqual(rec.state, ItemState.RESULT)
        self.assertTrue(rec.sealed)
        self.assertEqual(bundle.candidate_id, C1)

    def test_fp32_mismatch_not_silently_retried_here(self):
        runner = BoxRunner(
            config=PitConfig(enabled=True, mode='shadow'),
            io=ScriptedBoxIO(fail_master_digest=True),
            lane=LaneService(),
        )
        with self.assertRaises(Fp32MasterError):
            runner.run(self._item(), batch_handle='batch-handle:x', plan_handle='plan-handle')

    def test_bootstrap_override_refused(self):
        io = ScriptedBoxIO(train_result={
            'candidate_id': C1,
            'candidate_path': '/tmp/c',
            'candidate_master_path': '/tmp/m',
            'used_fp32_bootstrap_override': True,
        })
        runner = BoxRunner(config=PitConfig(enabled=True, mode='shadow'), io=io, lane=LaneService())
        with self.assertRaises(Fp32MasterError):
            runner.run(self._item(), batch_handle='batch-handle:x', plan_handle='plan-handle')

    def test_train_box_asserts(self):
        with self.assertRaises(TrainBoxError):
            assert_no_bootstrap_override({'AFFINE_FP32_BOOTSTRAP_OVERRIDE': '/tmp/x'})
        with self.assertRaises(TrainBoxError):
            run_train_box(
                train_fn=lambda **k: {'used_fp32_bootstrap_override': False},
                env={},
            )


if __name__ == '__main__':
    unittest.main()
