#!/usr/bin/env python3
"""Oldest-first rent: price failures keep the slot; newer items cannot jump."""
from __future__ import annotations

import unittest
from typing import Any, List

from . import PREFIX_SHADOW
from .config import PitConfig
from .controller import Controller
from .pe_lane import LaneService
from .pe_rent import RentError, RentService
from .state import ItemState


H0, M0 = 'a' * 64, 'b' * 64


class TestRentOldestFirst(unittest.TestCase):
    def test_price_failure_blocks_newer_and_keeps_attempt(self):
        calls: List[str] = []

        def precheck(**kwargs: Any):
            calls.append('precheck')
            raise RentError('offer price 80.00 above ceiling 56.00 (precheck)')

        rent = RentService(
            max_boxes=2,
            price_ceiling_usd_h=56.0,
            shadow=True,
            name_prefix=PREFIX_SHADOW,
            share_pe_cap=False,
            offer_precheck_fn=precheck,
        )
        ctl = Controller(
            config=PitConfig(enabled=True, mode='shadow', max_boxes=2),
            rent=rent,
            lane=LaneService(),
            box_io_factory=lambda: None,
            round_id_fn=lambda e: 'r-e%d' % int(e),
            plan_seed_fn=lambda e: 'p' * 64,
        )
        ctl.set_head(H0, epoch=1056, master_digest=M0)
        for ep, reg in ((1056, 1), (1057, 2), (1058, 3), (1060, 4)):
            ctl.enqueue(epoch=ep, registration_index=reg, hotkey_prefix='5M')

        out = ctl.tick()
        notes = [a.get('note') for a in out['actions'] if a.get('op') == 'advance']
        self.assertIn('rent-price-wait', notes)
        # Only the oldest waiter may attempt rent this tick.
        self.assertEqual(len(calls), 1)

        by_ep = {i.epoch: i for i in ctl.items}
        self.assertEqual(by_ep[1056].state, ItemState.RENTING)
        self.assertIsNone(by_ep[1056].box_id)
        self.assertEqual(by_ep[1056].attempt, 0)  # price failure must not burn attempts
        self.assertEqual(by_ep[1057].state, ItemState.BOUND)
        self.assertEqual(by_ep[1060].state, ItemState.QUEUED)  # max_boxes=2 → only 2 bound/waiting

        calls.clear()
        ctl.tick()
        self.assertEqual(len(calls), 1)
        self.assertEqual(by_ep[1057].state, ItemState.BOUND)
        self.assertIsNone(by_ep[1057].box_id)
        self.assertEqual(by_ep[1056].attempt, 0)

    def test_do_not_bind_beyond_free_slots_while_retry_wait(self):
        """RETRY_WAIT still occupies a slot so a newer item cannot bind ahead."""
        ctl = Controller(
            config=PitConfig(enabled=True, mode='shadow', max_boxes=1),
            rent=RentService(max_boxes=1, shadow=True, name_prefix=PREFIX_SHADOW),
            lane=LaneService(),
            box_io_factory=lambda: None,
            round_id_fn=lambda e: 'r-e%d' % int(e),
            plan_seed_fn=lambda e: 'p' * 64,
        )
        ctl.set_head(H0, epoch=1, master_digest=M0)
        older = ctl.enqueue(epoch=1056, registration_index=1, hotkey_prefix='5M')
        ctl.enqueue(epoch=1060, registration_index=2, hotkey_prefix='5M')
        older.state = ItemState.RETRY_WAIT
        older.last_error = 'rent:offer price 80.00 above ceiling 56.00'
        out = ctl.tick()
        bind_ops = [a for a in out['actions'] if a.get('op') == 'bound']
        self.assertEqual(bind_ops, [])
        newer = [i for i in ctl.items if i.epoch == 1060][0]
        self.assertEqual(newer.state, ItemState.QUEUED)


if __name__ == '__main__':
    unittest.main()
