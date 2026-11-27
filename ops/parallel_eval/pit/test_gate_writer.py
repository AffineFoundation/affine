#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from .gate_writer import (
    GateWriteError,
    RecordingGateWriter,
    validate_heldout,
)


def _full_heldout(**overrides):
    base = dict(
        round_id='r-test-1',
        baseline='prior-v1',
        set_size=500,
        bar_z=1.0,
        k_cand=8,
        head_rollouts=12,
        scoring='sn120-prior-baseline-v1',
        settings_hash='f' * 64,
        max_tokens=64000,
        scoring_mode='separate-run',
        tau_run=0.01,
        sigma_run_floor=0.05,
        baseline_done_at_cr='2026-01-01 00:00:00 CR',
    )
    base.update(overrides)
    return base


H, C = 'a' * 64, 'b' * 64


class TestGateWriter(unittest.TestCase):
    def test_recording_writes_full_heldout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gw = RecordingGateWriter(epochs_root=root)
            heldout = _full_heldout()
            path = gw.write_decision(
                epoch=1041,
                head=H,
                candidate=C,
                decision='green',
                evaluated=dict(z=1.2, n=400, gain=0.01),
                reasons=[],
                heldout=heldout,
                plan_seed='p' * 64,
                parallel_eval_meta=dict(source='pit', item_id='i1'),
            )
            self.assertTrue(path.exists())
            doc = json.loads(path.read_text())
            self.assertEqual(doc['heldout']['set_size'], 500)
            self.assertEqual(doc['heldout']['bar_z'], 1.0)
            self.assertTrue((root / '1041' / 'EVALUATED.private.json').exists())

    def test_refuses_missing_heldout_keys(self):
        with tempfile.TemporaryDirectory() as td:
            gw = RecordingGateWriter(epochs_root=Path(td))
            bad = _full_heldout()
            bad.pop('set_size')
            with self.assertRaises(GateWriteError):
                validate_heldout(bad)
            with self.assertRaises(GateWriteError):
                gw.write_decision(
                    epoch=1, head=H, candidate=C, decision='red',
                    evaluated=dict(z=None), reasons=['x'], heldout=bad,
                    plan_seed='p' * 64, parallel_eval_meta={},
                )

    def test_green_records_turn_add_and_close(self):
        with tempfile.TemporaryDirectory() as td:
            gw = RecordingGateWriter(epochs_root=Path(td))
            gw.write_decision(
                epoch=1, head=H, candidate=C, decision='green',
                evaluated=dict(z=1.5), reasons=[], heldout=_full_heldout(),
                plan_seed='p' * 64, parallel_eval_meta={},
            )
            self.assertEqual(len(gw.turn_adds), 1)
            self.assertEqual(gw.turn_adds[0][0], 'r-test-1')
            self.assertEqual(len(gw.closes), 1)

    def test_red_no_close(self):
        with tempfile.TemporaryDirectory() as td:
            gw = RecordingGateWriter(epochs_root=Path(td))
            gw.write_decision(
                epoch=1, head=H, candidate=C, decision='red',
                evaluated=dict(z=None), reasons=['fail'], heldout=_full_heldout(),
                plan_seed='p' * 64, parallel_eval_meta={},
            )
            self.assertEqual(len(gw.turn_adds), 1)
            self.assertEqual(gw.closes, [])

    def test_merge_blocked_green_no_close(self):
        with tempfile.TemporaryDirectory() as td:
            gw = RecordingGateWriter(epochs_root=Path(td))
            gw.write_decision(
                epoch=1, head=H, candidate=C, decision='green',
                evaluated=dict(z=1.0), reasons=['merge_blocked'], heldout=_full_heldout(),
                plan_seed='p' * 64, parallel_eval_meta={}, merge_blocked=True,
            )
            self.assertEqual(len(gw.turn_adds), 1)
            self.assertEqual(gw.closes, [])

    def test_shadow_refuse(self):
        with tempfile.TemporaryDirectory() as td:
            gw = RecordingGateWriter(epochs_root=Path(td), shadow=True)
            with self.assertRaises(GateWriteError):
                gw.write_decision(
                    epoch=1, head=H, candidate=C, decision='green',
                    evaluated=dict(z=1.0), reasons=[], heldout=_full_heldout(),
                    plan_seed='p' * 64, parallel_eval_meta={},
                )


if __name__ == '__main__':
    unittest.main()
