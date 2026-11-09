#!/usr/bin/env python3
"""Unit tests: hash-chained decision log and GREEN ordering."""
import tempfile
import unittest
from pathlib import Path

from .decision_log import GENESIS_PREV, DecisionLog


H0 = 'a' * 64
C1 = 'b' * 64
C2 = 'c' * 64


class TestDecisionLog(unittest.TestCase):
    def test_genesis_tip(self):
        log = DecisionLog()
        self.assertEqual(log.tip, GENESIS_PREV)
        self.assertEqual(len(log), 0)

    def test_hash_chain_links(self):
        log = DecisionLog()
        e0 = log.append(
            item_id='i0', registration_index=0, decision='red',
            bound_head=H0, paid=False, merge_blocked=False, shadow=True,
        )
        self.assertEqual(e0.prev_hash, GENESIS_PREV)
        e1 = log.append(
            item_id='i1', registration_index=1, decision='green',
            bound_head=H0, candidate_id=C1, z=1.4, paid=True,
            merge_blocked=False, green_at_mono=10.0, shadow=True,
        )
        self.assertEqual(e1.prev_hash, e0.entry_hash)
        e2 = log.append(
            item_id='i2', registration_index=2, decision='merge_blocked',
            bound_head=H0, candidate_id=C2, z=1.7, paid=True,
            merge_blocked=True, green_at_mono=20.0, shadow=True,
        )
        self.assertEqual(e2.prev_hash, e1.entry_hash)
        self.assertEqual([e.decision for e in log.greens_in_order()], ['green', 'merge_blocked'])

    def test_green_time_order_enforced(self):
        log = DecisionLog()
        log.append(
            item_id='i1', registration_index=1, decision='green',
            bound_head=H0, candidate_id=C1, paid=True, merge_blocked=False,
            green_at_mono=50.0, shadow=True,
        )
        with self.assertRaises(ValueError):
            log.append(
                item_id='i0', registration_index=0, decision='green',
                bound_head=H0, candidate_id=C2, paid=True, merge_blocked=False,
                green_at_mono=40.0, shadow=True,
            )

    def test_tamper_detected_on_load(self):
        log = DecisionLog()
        log.append(
            item_id='i1', registration_index=1, decision='green',
            bound_head=H0, candidate_id=C1, paid=True, merge_blocked=False,
            green_at_mono=1.0, shadow=True,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'log.json'
            log.save(path)
            # Tamper with a field but keep entry_hash — load must fail.
            import json
            doc = json.loads(path.read_text())
            doc['entries'][0]['z'] = 99.0
            path.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                DecisionLog.load(path)

    def test_roundtrip(self):
        log = DecisionLog()
        log.append(
            item_id='i1', registration_index=1, decision='merge_blocked',
            bound_head=H0, candidate_id=C1, z=1.2, paid=True,
            merge_blocked=True, green_at_mono=3.0, shadow=True,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'log.json'
            log.save(path)
            loaded = DecisionLog.load(path)
            self.assertEqual(loaded.tip, log.tip)
            self.assertEqual(loaded.entries[0].decision, 'merge_blocked')


if __name__ == '__main__':
    unittest.main()
