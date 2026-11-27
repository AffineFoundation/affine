#!/usr/bin/env python3
import unittest

from .state import (
    ALLOWED, RENTED_STATES, ItemRecord, ItemState, TransitionError,
    apply_transition, can_transition, next_states,
)


def queued(item_id='a' * 32):
    return ItemRecord(item_id=item_id, registration_index=1, epoch=100, hotkey_prefix='5ABC')


class TestStateMachine(unittest.TestCase):
    def test_every_state_listed(self):
        for state in ItemState:
            self.assertIn(state, ALLOWED)

    def test_happy_path_to_released(self):
        r = queued()
        for to in (
            ItemState.BOUND, ItemState.RENTING, ItemState.PROVISIONING,
            ItemState.TRAINING, ItemState.SEALED, ItemState.SERVING,
            ItemState.PROBING, ItemState.RESULT, ItemState.SCORED,
            ItemState.DECIDED, ItemState.RELEASED,
        ):
            r = apply_transition(r, to)
        self.assertEqual(r.state, ItemState.RELEASED)
        self.assertEqual(next_states(r), ())

    def test_illegal_edge(self):
        with self.assertRaises(TransitionError):
            apply_transition(queued(), ItemState.TRAINING)

    def test_rented_states_cover_live_box(self):
        self.assertIn(ItemState.PROBING, RENTED_STATES)
        self.assertNotIn(ItemState.QUEUED, RENTED_STATES)
        self.assertNotIn(ItemState.RELEASED, RENTED_STATES)

    def test_snapshot_has_no_secrets(self):
        blob = str(queued().snapshot()).lower()
        for banned in ('private_key', 'presign', 'secret', 'seed='):
            self.assertNotIn(banned, blob)

    def test_cannot_clobber_id(self):
        with self.assertRaises(TransitionError):
            apply_transition(queued(), ItemState.BOUND, updates={'item_id': 'x'})


if __name__ == '__main__':
    unittest.main()
