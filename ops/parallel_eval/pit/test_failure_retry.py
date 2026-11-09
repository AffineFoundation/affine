#!/usr/bin/env python3
"""Unit tests: failure / retry paths and fp32 fail-closed on the box."""
import unittest

from .adapters import stub_adapters
from .box import BoxRunner, Fp32MasterError, ScriptedBoxIO
from .config import PitConfig
from .dequeue import bind_item
from .host import HostController
from .state import ItemRecord, ItemState, apply_transition


H0 = 'a' * 64
M0 = 'b' * 64
C1 = 'c' * 64


class TestFailureRetry(unittest.TestCase):
    def test_box_fp32_mismatch_raises(self):
        cfg = PitConfig(enabled=True, mode='shadow')
        io = ScriptedBoxIO(fail_master_digest=True)
        runner = BoxRunner(config=cfg, io=io)
        rec = ItemRecord(item_id='x' * 32, registration_index=1, miner_hotkey_prefix='5A')
        rec = bind_item(rec, head=H0, head_epoch=1, master_digest=M0, dequeued_at_mono=1.0)
        rec = apply_transition(rec, ItemState.RENTING)
        rec = apply_transition(rec, ItemState.DOWNLOADING)
        with self.assertRaises(Fp32MasterError):
            runner.run(rec, batch_handle='batch-handle:x', plan_handle='plan-handle:1')

    def test_box_refuses_bootstrap_override_flag_in_train_receipt(self):
        cfg = PitConfig(enabled=True, mode='shadow')
        io = ScriptedBoxIO(train_result={
            'candidate_id': C1,
            'candidate_path': '/tmp/c',
            'candidate_master_path': '/tmp/cm',
            'used_fp32_bootstrap_override': True,
        })
        runner = BoxRunner(config=cfg, io=io)
        rec = ItemRecord(item_id='y' * 32, registration_index=1, miner_hotkey_prefix='5A')
        rec = bind_item(rec, head=H0, head_epoch=1, master_digest=M0, dequeued_at_mono=1.0)
        rec = apply_transition(rec, ItemState.RENTING)
        rec = apply_transition(rec, ItemState.DOWNLOADING)
        with self.assertRaises(Fp32MasterError):
            runner.run(rec, batch_handle='batch-handle:y', plan_handle='plan-handle:1')

    def test_host_retries_then_aborts(self):
        clock = {'t': 0.0}

        def now():
            clock['t'] += 1
            return float(clock['t'])

        def rent(**kwargs):
            raise RuntimeError('rent-failed')

        # Force adapter rent path (not shadow synthetic) so failures surface.
        ctl = HostController(
            config=PitConfig(enabled=True, mode='shadow', box_cap=2, max_infra_retries=1),
            adapters=stub_adapters(rent=rent, score=lambda **k: {'decision': 'red', 'z': 0.0}),
            box_io_factory=lambda: ScriptedBoxIO(),
            now_fn=now,
        )
        ctl.set_head(H0, epoch=1, master_digest=M0)
        ctl.enqueue(registration_index=1, miner_hotkey_prefix='5Miner')

        # ticks: bind, renting->fail->retry_wait, retry->renting->fail->abort-ish
        states = []
        for _ in range(12):
            ctl.tick()
            states.append(ctl.items[0].state.value)
            if ctl.items[0].state in (ItemState.ABORTED, ItemState.RELEASED, ItemState.FAILED):
                # One more tick may release/abort
                ctl.tick()
                states.append(ctl.items[0].state.value)
                break
        self.assertTrue(
            any(s in ('retry_wait', 'failed', 'aborted', 'released') for s in states),
            states,
        )
        self.assertGreaterEqual(ctl.items[0].attempt, 1)

    def test_gate_math_config_frozen(self):
        from .config import load_pit_config
        with self.assertRaises(ValueError):
            load_pit_config({'pit': {'enabled': True, 'mode': 'shadow', 'k_cand': 4}})
        with self.assertRaises(ValueError):
            load_pit_config({'pit': {'enabled': True, 'mode': 'shadow', 'dtype': 'fp8'}})
        with self.assertRaises(ValueError):
            load_pit_config({'pit': {'enabled': True, 'mode': 'live'}})


if __name__ == '__main__':
    unittest.main()
