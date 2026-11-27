"""Live HostGateWriter through the real heldout.Rounds layout (not shadow)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from miner.validator.gate.heldout import Rounds

from .gate_writer import (
    GateWriteError,
    HostGateWriter,
    _wj,
    _rj,
    _log,
    _cr,
    build_host_rounds,
)
from .test_gate_writer import H, C, _full_heldout


class TestHostGateWriterLive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.gate = self.root / 'gate'
        self.epochs = self.gate / 'epochs'
        self.heldout = self.gate / 'heldout'
        self.epochs.mkdir(parents=True)
        self.heldout.mkdir(parents=True)
        self.rounds = Rounds(self.heldout, _wj, _rj, _log, _cr)
        # Minimal open round so turn_add / close have a directory.
        rid = 'r-live-1'
        (self.heldout / rid).mkdir()
        _wj(self.heldout / rid / 'ROUND.private.json', dict(id=rid, head=H, n=1), 0o600)
        self.rid = rid
        self.addCleanup(self.tmp.cleanup)
        self.env = {
            'HG_GATE_STATE': str(self.gate),
            'PE_GATE_EP': str(self.epochs),
            'PE_HELDOUT': str(self.heldout),
        }

    def _writer(self):
        return HostGateWriter(
            epochs_root=self.epochs,
            shadow=False,
            rounds=self.rounds,
            k1_grant_fn=lambda e, why: 'granted-test',
        )

    def test_red_turn_add_no_close(self):
        gw = self._writer()
        path = gw.write_decision(
            epoch=1055, head=H, candidate=C, decision='red',
            evaluated=dict(z=-0.5), reasons=['below_bar'],
            heldout=_full_heldout(round_id=self.rid),
            plan_seed='p' * 64, parallel_eval_meta=dict(source='pit'),
        )
        self.assertTrue(path.exists())
        turns = self.rounds.turns(self.rid)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]['decision'], 'red')
        self.assertEqual(turns[0]['epoch'], 1055)
        self.assertFalse(self.rounds.closed(self.rid))

    def test_green_turn_add_and_close(self):
        gw = self._writer()
        path = gw.write_decision(
            epoch=1056, head=H, candidate=C, decision='green',
            evaluated=dict(z=1.5), reasons=[],
            heldout=_full_heldout(round_id=self.rid),
            plan_seed='p' * 64, parallel_eval_meta=dict(source='pit'),
        )
        self.assertTrue(path.exists())
        turns = self.rounds.turns(self.rid)
        self.assertEqual(turns[-1]['decision'], 'green')
        self.assertTrue(self.rounds.closed(self.rid))

    def test_missing_rounds_writes_nothing(self):
        gw = HostGateWriter(
            epochs_root=self.epochs,
            shadow=False,
            rounds=None,
            k1_grant_fn=lambda e, why: None,
        )
        with mock.patch.dict(os.environ, {
            'HG_GATE_STATE': '', 'PE_GATE_STATE': '', 'PE_GATE_EP': '', 'PE_HELDOUT': '',
        }, clear=False):
            with mock.patch('ops.parallel_eval.pit.gate_writer._import_gate_service', return_value=None):
                with mock.patch('ops.parallel_eval.pit.gate_writer.importlib.import_module', side_effect=ImportError('x')):
                    with self.assertRaises(GateWriteError):
                        gw.write_decision(
                            epoch=1, head=H, candidate=C, decision='red',
                            evaluated=dict(z=None), reasons=['x'],
                            heldout=_full_heldout(round_id=self.rid),
                            plan_seed='p' * 64, parallel_eval_meta={},
                        )
        self.assertFalse((self.epochs / '1' / 'DECISION.json').exists())
        self.assertFalse((self.epochs / '1' / 'EVALUATED.private.json').exists())

    def test_idempotent_rewrite_identical(self):
        gw = self._writer()
        heldout = _full_heldout(round_id=self.rid)
        p1 = gw.write_decision(
            epoch=10, head=H, candidate=C, decision='red',
            evaluated=dict(z=0.1), reasons=['a'], heldout=heldout,
            plan_seed='p' * 64, parallel_eval_meta={},
        )
        p2 = gw.write_decision(
            epoch=10, head=H, candidate=C, decision='red',
            evaluated=dict(z=0.1), reasons=['a'], heldout=heldout,
            plan_seed='p' * 64, parallel_eval_meta={},
        )
        self.assertEqual(p1, p2)
        # Duplicate turn rows suppressed.
        turns = [t for t in self.rounds.turns(self.rid) if t.get('epoch') == 10]
        self.assertEqual(len(turns), 1)

    def test_conflicting_rewrite_refuses(self):
        gw = self._writer()
        heldout = _full_heldout(round_id=self.rid)
        gw.write_decision(
            epoch=11, head=H, candidate=C, decision='red',
            evaluated=dict(z=0.1), reasons=['a'], heldout=heldout,
            plan_seed='p' * 64, parallel_eval_meta={},
        )
        with self.assertRaisesRegex(GateWriteError, 'conflicting'):
            gw.write_decision(
                epoch=11, head=H, candidate=C, decision='green',
                evaluated=dict(z=2.0), reasons=[], heldout=heldout,
                plan_seed='p' * 64, parallel_eval_meta={},
            )

    def test_build_host_rounds_from_env(self):
        with mock.patch.dict(os.environ, self.env, clear=False):
            R = build_host_rounds()
            self.assertEqual(Path(R.root), self.heldout)


if __name__ == '__main__':
    unittest.main()
