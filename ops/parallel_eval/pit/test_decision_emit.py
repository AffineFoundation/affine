#!/usr/bin/env python3
"""Gate-schema DECISION.json emit for k1 / ff_launcher / pe_lifecycle."""
import json
import tempfile
import unittest
from pathlib import Path

from priors import baseline_v1 as BV

from .config import load_pit_config
from .decision_emit import build_decision_doc, emit_decision, map_pit_decision
from .gate_writer import GateWriteError, RecordingGateWriter
from .state import ItemRecord, ItemState


def _heldout():
    return dict(
        round_id='round-1',
        baseline='prior-v1',
        set_size=500,
        bar_z=1.0,
        k_cand=8,
        head_rollouts=12,
        scoring=BV.VERSION,
        settings_hash='d' * 64,
        max_tokens=64000,
        scoring_mode='separate-run',
        tau_run=0.01,
        sigma_run_floor=0.05,
        baseline_done_at_cr='2026-01-01 00:00:00 CR',
    )


class TestDecisionEmit(unittest.TestCase):
    def _item(self, **kw):
        base = dict(
            item_id='item1', registration_index=0, epoch=1041,
            uid=7, hotkey_prefix='hk', state=ItemState.DECIDED,
            bound_head='a' * 64, bound_round_id='round-1',
            candidate_id='b' * 64, decision='green_fresh', z=1.5,
            fresh=True, merge_blocked=False, paid=True, shadow=False,
        )
        base.update(kw)
        return ItemRecord(**base)

    def test_map_decisions(self):
        self.assertEqual(map_pit_decision('green_fresh'), 'green')
        self.assertEqual(map_pit_decision('green_stale'), 'green')
        self.assertEqual(map_pit_decision('red'), 'red')
        self.assertEqual(map_pit_decision('no_decision'), 'no_decision')

    def test_build_schema_fields(self):
        item = self._item()
        scored = {
            'evaluated': {'z': 1.5, 'n': 1000, 'c_pass': 0.5, 'h_pass': 0.4},
            'reasons': [],
            'heldout': {'round_id': 'round-1'},
        }
        doc = build_decision_doc(item, scored=scored, mode='live')
        self.assertEqual(doc['decision'], 'green')
        self.assertEqual(doc['source'], 'pit')
        self.assertEqual(doc['epoch'], 1041)
        self.assertEqual(doc['candidate'], 'b' * 64)
        self.assertEqual(doc['head'], 'a' * 64)
        self.assertEqual(doc['red_credit'], 'zero')
        self.assertEqual(doc['parallel_eval']['source'], 'pit')
        self.assertIn('evaluated', doc)
        self.assertFalse(doc['no_decision'])

    def test_shadow_refuses_write(self):
        cfg = load_pit_config({'pit': {'enabled': True, 'mode': 'shadow'}})
        with tempfile.TemporaryDirectory() as td:
            gw = RecordingGateWriter(epochs_root=Path(td))
            path = emit_decision(
                self._item(shadow=True, bound_plan_seed='e' * 64),
                scored={'evaluated': {'z': 1.0}, 'heldout': _heldout()},
                gate_writer=gw,
                config=cfg,
            )
            self.assertIsNone(path)
            self.assertEqual(list(Path(td).glob('**/DECISION.json')), [])

    def test_live_writes_decision(self):
        cfg = load_pit_config({
            'pit': {'enabled': True, 'mode': 'live', 'cutover': True, 'max_boxes': 5},
        })
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gw = RecordingGateWriter(epochs_root=root)
            item = self._item(bound_plan_seed='e' * 64)
            path = emit_decision(
                item,
                scored={'evaluated': {'z': 1.5, 'n': 1000}, 'reasons': [], 'heldout': _heldout()},
                gate_writer=gw,
                config=cfg,
            )
            self.assertIsNotNone(path)
            doc = json.loads(path.read_text())
            self.assertEqual(doc['decision'], 'green')
            self.assertEqual(doc['source'], 'pit')
            self.assertEqual(doc['parallel_eval']['item_id'], 'item1')
            self.assertEqual(len(gw.turn_adds), 1)

    def test_emit_refuses_incomplete_heldout(self):
        cfg = load_pit_config({
            'pit': {'enabled': True, 'mode': 'live', 'cutover': True, 'max_boxes': 5},
        })
        with tempfile.TemporaryDirectory() as td:
            gw = RecordingGateWriter(epochs_root=Path(td))
            with self.assertRaises(GateWriteError):
                emit_decision(
                    self._item(bound_plan_seed='e' * 64),
                    scored={'evaluated': {'z': 1.0}, 'heldout': {'round_id': 'r1'}},
                    gate_writer=gw,
                    config=cfg,
                )

    def test_red_no_decision_flags(self):
        item = self._item(decision='no_decision', paid=False, fresh=None, z=None)
        doc = build_decision_doc(item, scored={})
        self.assertEqual(doc['decision'], 'no_decision')
        self.assertTrue(doc['no_decision'])
        self.assertEqual(doc['chart_status'], 'no_decision')


if __name__ == '__main__':
    unittest.main()
