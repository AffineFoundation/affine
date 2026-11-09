#!/usr/bin/env python3
"""Unit tests: dequeue binding and staleness / merge_blocked rules."""
import unittest

from .dequeue import (
    bind_item,
    is_stale_green,
    mark_green_outcome,
    mark_red_outcome,
    order_greens,
    select_next_queued,
)
from .state import ItemRecord, ItemState, TransitionError, apply_transition


H0 = 'a' * 64
H1 = 'b' * 64
M0 = 'c' * 64
C1 = 'd' * 64
C2 = 'e' * 64


def queued(reg, item_id=None):
    return ItemRecord(
        item_id=item_id or ('i%d' % reg).ljust(32, '0'),
        registration_index=reg,
        miner_hotkey_prefix='5ABC',
    )


class TestBinding(unittest.TestCase):
    def test_bind_stamps_head_and_master(self):
        r = bind_item(queued(1), head=H0, head_epoch=10, master_digest=M0, dequeued_at_mono=1.0)
        self.assertEqual(r.state, ItemState.BOUND)
        self.assertEqual(r.bound_head, H0)
        self.assertEqual(r.bound_master_digest, M0)
        self.assertEqual(r.bound_head_epoch, 10)

    def test_bind_rejects_non_queued(self):
        r = bind_item(queued(1), head=H0, head_epoch=1, master_digest=M0, dequeued_at_mono=1.0)
        with self.assertRaises(TransitionError):
            bind_item(r, head=H0, head_epoch=1, master_digest=M0, dequeued_at_mono=2.0)

    def test_first_green_applies(self):
        r = bind_item(queued(1), head=H0, head_epoch=1, master_digest=M0, dequeued_at_mono=1.0)
        for to in (ItemState.RENTING, ItemState.DOWNLOADING, ItemState.TRAINING,
                   ItemState.PROBING, ItemState.SCORING):
            r = apply_transition(r, to)
        out = mark_green_outcome(r, current_head=H0, z=1.5, candidate_id=C1)
        self.assertEqual(out.state, ItemState.APPLIED)
        self.assertTrue(out.paid)
        self.assertFalse(out.merge_blocked)

    def test_later_green_on_old_head_is_merge_blocked(self):
        r = bind_item(queued(2), head=H0, head_epoch=1, master_digest=M0, dequeued_at_mono=2.0)
        for to in (ItemState.RENTING, ItemState.DOWNLOADING, ItemState.TRAINING,
                   ItemState.PROBING, ItemState.SCORING):
            r = apply_transition(r, to)
        # Head already moved to H1 (first GREEN applied elsewhere).
        out = mark_green_outcome(r, current_head=H1, z=2.0, candidate_id=C2)
        self.assertEqual(out.state, ItemState.MERGE_BLOCKED)
        self.assertTrue(out.paid)
        self.assertTrue(out.merge_blocked)
        self.assertTrue(is_stale_green(out, H1))

    def test_red_not_paid(self):
        r = bind_item(queued(1), head=H0, head_epoch=1, master_digest=M0, dequeued_at_mono=1.0)
        for to in (ItemState.RENTING, ItemState.DOWNLOADING, ItemState.TRAINING,
                   ItemState.PROBING, ItemState.SCORING):
            r = apply_transition(r, to)
        out = mark_red_outcome(r, z=0.2)
        self.assertEqual(out.state, ItemState.RED)
        self.assertFalse(out.paid)
        self.assertFalse(out.merge_blocked)

    def test_order_greens_by_green_at(self):
        a = queued(1, 'a' * 32)
        b = queued(2, 'b' * 32)
        a = bind_item(a, head=H0, head_epoch=1, master_digest=M0, dequeued_at_mono=1.0)
        b = bind_item(b, head=H0, head_epoch=1, master_digest=M0, dequeued_at_mono=2.0)
        for item, cand, gat in ((a, C1, 200.0), (b, C2, 100.0)):
            r = item
            for to in (ItemState.RENTING, ItemState.DOWNLOADING, ItemState.TRAINING,
                       ItemState.PROBING, ItemState.SCORING):
                r = apply_transition(r, to)
            r = mark_green_outcome(r, current_head=H0 if gat == 100 else C2, z=1.1, candidate_id=cand)
            r.meta['green_at_mono'] = gat
            if item is a:
                a = r
            else:
                b = r
        # b finished earlier (100) even though registered later; a is later green (200) and may be merge_blocked
        ordered = order_greens([a, b])
        self.assertEqual([x.item_id for x in ordered], [b.item_id, a.item_id])

    def test_box_cap_blocks_dequeue(self):
        items = [queued(1), queued(2), queued(3)]
        self.assertIsNotNone(select_next_queued(items, in_flight=0, box_cap=2))
        self.assertIsNone(select_next_queued(items, in_flight=2, box_cap=2))
        self.assertEqual(select_next_queued(items, in_flight=1, box_cap=2).registration_index, 1)


if __name__ == '__main__':
    unittest.main()
