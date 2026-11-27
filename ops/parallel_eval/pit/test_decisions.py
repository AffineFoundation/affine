#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

from .decisions import GENESIS_PREV, DecisionLog


H0, C1, C2 = 'a' * 64, 'b' * 64, 'c' * 64


class TestDecisions(unittest.TestCase):
    def test_chain(self):
        log = DecisionLog()
        self.assertEqual(log.tip, GENESIS_PREV)
        e0 = log.append(
            epoch=1, uid=1, hotkey_prefix='5A', binding_H=H0, binding_round_id='r1',
            candidate=C1, decision='red', paid=False, decided_at=10.0,
        )
        e1 = log.append(
            epoch=2, uid=2, hotkey_prefix='5B', binding_H=H0, binding_round_id='r1',
            candidate=C2, decision='green_fresh', z=1.4, paid=True, fresh=True,
            merge_blocked=False, decided_at=20.0,
        )
        self.assertEqual(e1.prev_sha256, e0.entry_hash)
        self.assertEqual([e.decision for e in log.greens_in_order()], ['green_fresh'])

    def test_green_order_enforced(self):
        log = DecisionLog()
        log.append(
            epoch=1, uid=1, hotkey_prefix='5A', binding_H=H0, binding_round_id='r',
            candidate=C1, decision='green_fresh', paid=True, fresh=True, decided_at=50.0,
        )
        with self.assertRaises(ValueError):
            log.append(
                epoch=2, uid=2, hotkey_prefix='5B', binding_H=H0, binding_round_id='r',
                candidate=C2, decision='green_stale', paid=True, fresh=False,
                merge_blocked=True, decided_at=40.0,
            )

    def test_public_slice_has_no_indices(self):
        log = DecisionLog()
        log.append(
            epoch=1, uid=1, hotkey_prefix='5A', binding_H=H0, binding_round_id='r',
            candidate=C1, decision='green_fresh', paid=True, fresh=True, decided_at=1.0,
            extra={'note': 'x'},
        )
        pub = log.public_slice()[0]
        blob = str(pub).lower()
        self.assertNotIn('indices', blob)
        self.assertNotIn('seed', blob)

    def test_tamper_detected(self):
        log = DecisionLog()
        log.append(
            epoch=1, uid=1, hotkey_prefix='5A', binding_H=H0, binding_round_id='r',
            candidate=C1, decision='green_fresh', paid=True, fresh=True, decided_at=1.0,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'log.json'
            log.save(path)
            import json
            doc = json.loads(path.read_text())
            doc['entries'][0]['z'] = 99.0
            path.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                DecisionLog.load(path)

    def test_fresh_or_stale(self):
        log = DecisionLog()
        self.assertEqual(log.fresh_or_stale(H0, H0), 'fresh')
        self.assertEqual(log.fresh_or_stale(H0, C1), 'stale')


if __name__ == '__main__':
    unittest.main()
