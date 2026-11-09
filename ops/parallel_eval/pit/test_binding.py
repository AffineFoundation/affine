#!/usr/bin/env python3
import unittest

from .binding import (
    dequeue_and_bind, is_stale_green, mark_scored_decision, order_greens,
    rebind_if_unsealed, select_next_queued,
)
from .state import ItemRecord, ItemState, TransitionError, apply_transition


H0, H1, M0, C1, C2 = 'a' * 64, 'b' * 64, 'c' * 64, 'd' * 64, 'e' * 64


def queued(reg, epoch=None):
    return ItemRecord(
        item_id=('i%d' % reg).ljust(32, '0'),
        registration_index=reg,
        epoch=epoch if epoch is not None else 1000 + reg,
        hotkey_prefix='5ABC',
    )


class TestBinding(unittest.TestCase):
    def test_bind_stamps_head_and_master(self):
        r = dequeue_and_bind(
            queued(1), head=H0, head_epoch=10, round_id='r1',
            round_seed_commitment='commit1', master_digest=M0, dequeued_at_mono=1.0,
        )
        self.assertEqual(r.state, ItemState.BOUND)
        self.assertEqual(r.bound_head, H0)
        self.assertEqual(r.bound_master_digest, M0)
        self.assertEqual(r.bound_round_id, 'r1')
        self.assertNotIn('indices', r.meta.get('binding', {}))

    def test_dequeue_pauses_on_unpublished_master(self):
        r = dequeue_and_bind(
            queued(1), head=H0, head_epoch=1, round_id='r1',
            round_seed_commitment='c', master_digest=None,
            public_manifest={},  # empty → lookup fails
        )
        self.assertIsNone(r)

    def test_first_green_fresh(self):
        r = dequeue_and_bind(
            queued(1), head=H0, head_epoch=1, round_id='r1',
            round_seed_commitment='c', master_digest=M0, dequeued_at_mono=1.0,
        )
        for to in (ItemState.RENTING, ItemState.PROVISIONING, ItemState.TRAINING,
                   ItemState.SEALED, ItemState.SERVING, ItemState.PROBING,
                   ItemState.RESULT, ItemState.SCORED):
            r = apply_transition(r, to)
        out = mark_scored_decision(r, current_head=H0, decision='green', z=1.5, candidate_id=C1)
        self.assertEqual(out.decision, 'green_fresh')
        self.assertFalse(out.merge_blocked)
        self.assertTrue(out.paid)

    def test_stale_green_merge_blocked(self):
        r = dequeue_and_bind(
            queued(2), head=H0, head_epoch=1, round_id='r1',
            round_seed_commitment='c', master_digest=M0, dequeued_at_mono=2.0,
        )
        for to in (ItemState.RENTING, ItemState.PROVISIONING, ItemState.TRAINING,
                   ItemState.SEALED, ItemState.SERVING, ItemState.PROBING,
                   ItemState.RESULT, ItemState.SCORED):
            r = apply_transition(r, to)
        out = mark_scored_decision(r, current_head=H1, decision='green', z=2.0, candidate_id=C2)
        self.assertEqual(out.decision, 'green_stale')
        self.assertTrue(out.merge_blocked)
        self.assertTrue(out.paid)
        self.assertTrue(is_stale_green(out, H1))

    def test_rebind_only_before_sealed(self):
        r = dequeue_and_bind(
            queued(1), head=H0, head_epoch=1, round_id='r1',
            round_seed_commitment='c', master_digest=M0, dequeued_at_mono=1.0,
        )
        r = apply_transition(r, ItemState.RENTING)
        r = rebind_if_unsealed(
            r, head=H1, head_epoch=2, round_id='r2',
            round_seed_commitment='c2', master_digest=M0,
        )
        self.assertEqual(r.bound_head, H1)
        r.sealed = True
        with self.assertRaises(TransitionError):
            rebind_if_unsealed(
                r, head=H0, head_epoch=3, round_id='r3',
                round_seed_commitment='c3', master_digest=M0,
            )

    def test_box_cap_blocks_dequeue(self):
        items = [queued(1), queued(2), queued(3)]
        self.assertIsNotNone(select_next_queued(items, rented=0, max_boxes=2))
        self.assertIsNone(select_next_queued(items, rented=2, max_boxes=2))


if __name__ == '__main__':
    unittest.main()
