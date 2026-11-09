#!/usr/bin/env python3
"""Reproduce /tmp/pitsim review blockers: no fabricated GREEN DECISION; no early emit."""
import json
import tempfile
import unittest
from pathlib import Path

from .apply_live import LiveApplicator
from .box import ScriptedBoxIO
from .config import PitConfig, load_pit_config
from .controller import Controller
from .durable import DurableStore
from .gate_writer import RecordingGateWriter
from .pe_lane import LaneService
from .pe_rent import RentService
from .run_host import _gate_emit_ready, tick_once
from .state import ItemState


H0, M0 = 'a' * 64, 'b' * 64


def _heldout():
    return {
        'round_id': 'round-1',
        'set_size': 1000,
        'bar_z': 1.0,
        'k_cand': 8,
        'head_rollouts': 12,
        'scoring': 'baseline_v1',
        'settings_hash': 's' * 64,
        'baseline': 'prior-v1',
        'max_tokens': 65536,
        'scoring_mode': 'separate',
        'tau_run': 0.0,
        'sigma_run_floor': 0.0,
        'baseline_done_at_cr': 'x',
    }


class TestPitsimBlockers(unittest.TestCase):
    def test_live_from_flag_refuses_scripted_default(self):
        with self.assertRaises(ValueError):
            Controller.from_flag({'pit': {'enabled': True, 'mode': 'live', 'cutover': True, 'max_boxes': 5}})

    def test_green_decision_not_emitted_before_verify(self):
        cfg = PitConfig(enabled=True, mode='live', max_boxes=1, cutover=True)
        item_states = []

        def train(**kwargs):
            return {
                'candidate_id': 'c' * 64,
                'candidate_path': '/tmp/c',
                'candidate_master_path': '/tmp/m',
                'candidate_master_digest': M0,
                'candidate_sha256': 'c' * 64,
                'used_fp32_bootstrap_override': False,
            }

        applicator = LiveApplicator(
            config=cfg,
            verify_uploads=lambda **k: {'candidate_ok': False, 'master_ok': True},
            accept_pit_decision=lambda p: (_ for _ in ()).throw(RuntimeError('should not apply')),
        )
        ctl = Controller(
            config=cfg,
            rent=RentService(max_boxes=1),
            lane=LaneService(),
            box_io_factory=lambda: ScriptedBoxIO(on_train=train),
            score_fn=lambda **k: {
                'decision': 'green', 'z': 1.5, 'evaluated': {'z': 1.5},
                'heldout': _heldout(),
            },
            live_applicator=applicator,
        )
        ctl.set_head(H0, epoch=1, master_digest=M0)
        ctl.enqueue(epoch=42, registration_index=1)
        for _ in range(30):
            ctl.tick()
            for it in ctl.items:
                item_states.append((it.state.value, it.decision, _gate_emit_ready(it)))
            if ctl.items and ctl.items[0].state is ItemState.RELEASED:
                break
        # Never ready to emit a paid green; final decision is no_decision unpaid.
        self.assertFalse(any(
            ready and dec == 'green_fresh' for _st, dec, ready in item_states
        ))
        self.assertEqual(ctl.items[0].decision, 'no_decision')
        self.assertFalse(ctl.items[0].paid)
        self.assertEqual(ctl.current_head, H0)

    def test_tick_once_skips_emit_without_verified_apply(self):
        cfg = load_pit_config({'pit': {'enabled': True, 'mode': 'live', 'cutover': True, 'max_boxes': 1}})
        with tempfile.TemporaryDirectory() as td:
            store = DurableStore(root=Path(td))
            ctl = Controller(
                config=cfg,
                rent=RentService(max_boxes=1),
                lane=LaneService(),
                box_io_factory=lambda: ScriptedBoxIO(),
                score_fn=lambda **k: {'decision': 'green', 'z': 1.5, 'heldout': _heldout()},
            )
            ctl.set_head(H0, epoch=1, master_digest=M0)
            item = ctl.enqueue(epoch=7, registration_index=0)
            item.decision = 'green_fresh'
            item.paid = True
            item.candidate_id = 'c' * 64
            item.bound_head = H0
            item.meta = {'scored': {'decision': 'green', 'heldout': _heldout(), 'evaluated': {'z': 1.5}}}
            # No live_apply → must not emit.
            self.assertFalse(_gate_emit_ready(item))
            out = tick_once(ctl, store, dry_run=False)
            self.assertEqual(out.get('advanced') or [], [])
            self.assertFalse((Path(td) / 'items').joinpath('%s.json' % item.item_id).read_text().count('decision_emitted'))


if __name__ == '__main__':
    unittest.main()
