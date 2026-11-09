#!/usr/bin/env python3
"""Unit tests: PIT item state machine."""
import unittest

from .state import (
    ALLOWED,
    ItemRecord,
    ItemState,
    TransitionError,
    apply_transition,
    can_transition,
    next_states,
)


def queued(item_id='a' * 32):
    return ItemRecord(item_id=item_id, registration_index=1, miner_hotkey_prefix='5ABC')


class TestStateMachine(unittest.TestCase):
    def test_every_state_has_an_allowed_entry(self):
        for state in ItemState:
            self.assertIn(state, ALLOWED)

    def test_happy_path_to_applied(self):
        r = queued()
        for to in (
            ItemState.BOUND, ItemState.RENTING, ItemState.DOWNLOADING,
            ItemState.TRAINING, ItemState.PROBING, ItemState.SCORING,
            ItemState.GREEN, ItemState.APPLIED, ItemState.RELEASED,
        ):
            r = apply_transition(r, to)
            self.assertEqual(r.state, to)

    def test_happy_path_to_merge_blocked(self):
        r = queued()
        for to in (
            ItemState.BOUND, ItemState.RENTING, ItemState.DOWNLOADING,
            ItemState.TRAINING, ItemState.PROBING, ItemState.SCORING,
            ItemState.GREEN, ItemState.MERGE_BLOCKED, ItemState.RELEASED,
        ):
            r = apply_transition(r, to)
        self.assertEqual(r.state, ItemState.RELEASED)

    def test_red_path(self):
        r = queued()
        for to in (
            ItemState.BOUND, ItemState.RENTING, ItemState.DOWNLOADING,
            ItemState.TRAINING, ItemState.PROBING, ItemState.SCORING,
            ItemState.RED, ItemState.RELEASED,
        ):
            r = apply_transition(r, to)
        self.assertEqual(r.state, ItemState.RELEASED)

    def test_retry_path(self):
        r = queued()
        r = apply_transition(r, ItemState.BOUND)
        r = apply_transition(r, ItemState.RENTING)
        r = apply_transition(r, ItemState.RETRY_WAIT, reason='rent-timeout')
        self.assertEqual(r.last_error, 'rent-timeout')
        r = apply_transition(r, ItemState.RENTING)
        self.assertEqual(r.state, ItemState.RENTING)

    def test_illegal_edge_raises(self):
        r = queued()
        with self.assertRaises(TransitionError):
            apply_transition(r, ItemState.TRAINING)
        with self.assertRaises(TransitionError):
            apply_transition(r, ItemState.GREEN)

    def test_released_is_terminal(self):
        r = apply_transition(queued(), ItemState.ABORTED)
        r = apply_transition(r, ItemState.RELEASED)
        self.assertEqual(next_states(r), ())
        self.assertFalse(can_transition(r, ItemState.QUEUED))

    def test_updates_cannot_clobber_id_or_state(self):
        r = queued()
        with self.assertRaises(TransitionError):
            apply_transition(r, ItemState.BOUND, updates={'item_id': 'x'})
        with self.assertRaises(TransitionError):
            apply_transition(r, ItemState.BOUND, updates={'state': ItemState.GREEN})

    def test_snapshot_has_no_secret_fields(self):
        r = queued()
        snap = r.snapshot()
        blob = str(snap).lower()
        for banned in ('private_key', 'presign', 'secret', 'seed='):
            self.assertNotIn(banned, blob)


if __name__ == '__main__':
    unittest.main()
