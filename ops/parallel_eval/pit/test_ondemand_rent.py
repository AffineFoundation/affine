#!/usr/bin/env python3
"""Hard rule: on-demand rent only. Empty queue ⇒ 0 rentals; B queued, cap N ⇒ min(B,N)."""
import unittest

from .box import ScriptedBoxIO
from .config import PitConfig, load_pit_config
from .controller import Controller
from .pe_rent import RentService
from .pe_lane import LaneService
from .state import ItemState


H0, M0 = 'a' * 64, 'b' * 64


class TestOnDemandRent(unittest.TestCase):
    def _ctl(self, max_boxes=2, score=None):
        clock = {'t': 100.0}

        def now():
            clock['t'] += 1.0
            return clock['t']

        cand = {'n': 0}

        def train(**kwargs):
            cand['n'] += 1
            return {
                'candidate_id': ('%x' % cand['n']).rjust(64, '0'),
                'candidate_path': '/tmp/c%d' % cand['n'],
                'candidate_master_path': '/tmp/m%d' % cand['n'],
                'used_fp32_bootstrap_override': False,
            }

        return Controller(
            config=PitConfig(enabled=True, mode='shadow', max_boxes=max_boxes),
            rent=RentService(max_boxes=max_boxes),
            lane=LaneService(),
            box_io_factory=lambda: ScriptedBoxIO(on_train=train),
            now_fn=now,
            score_fn=score or (lambda **k: {'decision': 'green', 'z': 1.5}),
            # Hermetic: SAMPLE.private.json round_id (no fixture default).
            round_id_fn=lambda e: 'r-e%d' % int(e),
            plan_seed_fn=lambda e: ('p' * 64),
        )

    def test_empty_queue_zero_rentals(self):
        ctl = self._ctl()
        ctl.set_head(H0, epoch=1, master_digest=M0)
        out = ctl.tick()
        self.assertEqual(out['rentals'], 0)
        self.assertEqual(ctl.rented_count(), 0)
        self.assertEqual(ctl.rent.ledger.count(), 0)

    def test_b_queued_cap_n_means_min_b_n_rentals(self):
        for B, N, expect in ((1, 2, 1), (2, 2, 2), (5, 2, 2), (3, 3, 3)):
            ctl = self._ctl(max_boxes=N)
            ctl.set_head(H0, epoch=1, master_digest=M0)
            for i in range(B):
                ctl.enqueue(epoch=100 + i, registration_index=i + 1, hotkey_prefix='5M')
            # One tick binds + rents up to cap (does not necessarily finish items).
            ctl.tick()
            self.assertEqual(
                ctl.rented_count(), expect,
                'B=%d N=%d expected %d rentals, got %d' % (B, N, expect, ctl.rented_count()),
            )
            # Never more than N, never more than B.
            self.assertLessEqual(ctl.rented_count(), N)
            self.assertLessEqual(ctl.rented_count(), B)

    def test_release_on_result_or_fail_drops_rental(self):
        ctl = self._ctl(max_boxes=2)
        ctl.set_head(H0, epoch=1, master_digest=M0)
        ctl.enqueue(epoch=10, registration_index=1, hotkey_prefix='5M')
        for _ in range(20):
            ctl.tick()
            if ctl.items[0].state is ItemState.RELEASED:
                break
        self.assertEqual(ctl.items[0].state, ItemState.RELEASED)
        self.assertEqual(ctl.rented_count(), 0)
        self.assertIsNone(ctl.items[0].box_id)

    def test_warm_pool_config_refused(self):
        with self.assertRaises(ValueError):
            load_pit_config({'pit': {'enabled': True, 'mode': 'shadow', 'warm_pool': 2}})
        with self.assertRaises(ValueError):
            load_pit_config({'pit': {'enabled': True, 'mode': 'shadow', 'pre_rent': True}})

    def test_no_rent_without_demand(self):
        rent = RentService(max_boxes=2)
        self.assertFalse(rent.can_rent(queued_bound_waiting=0))
        self.assertTrue(rent.can_rent(queued_bound_waiting=1))


if __name__ == '__main__':
    unittest.main()
