#!/usr/bin/env python3
import unittest

from priors import baseline_v1 as BV

from .binding import dequeue_and_bind, live_cand_seed
from .state import ItemRecord, ItemState


class TestPlanSeed(unittest.TestCase):
    def test_live_cand_seed_matches_baseline_v1(self):
        seed = 'reveal-seed-abc'
        for epoch in (0, 507, 1041):
            self.assertEqual(live_cand_seed(seed, epoch), BV.cand_seed(seed, epoch))

    def test_item_record_meta_never_stores_raw_secret(self):
        item = ItemRecord(
            item_id='item-secret-test',
            registration_index=0,
            epoch=1041,
            hotkey_prefix='hk',
        )
        secret = 'super-secret-round-seed-value'
        bound = dequeue_and_bind(
            item,
            head='a' * 64,
            head_epoch=10,
            round_id='r1',
            secret_seed=secret,
            master_digest='c' * 64,
        )
        self.assertEqual(bound.state, ItemState.BOUND)
        self.assertEqual(bound.bound_plan_seed, live_cand_seed(secret, 1041))
        meta_blob = str(bound.meta)
        self.assertNotIn(secret, meta_blob)
        binding = bound.meta.get('binding') or {}
        self.assertEqual(binding.get('plan_seed'), bound.bound_plan_seed)
        self.assertNotIn('secret_seed', binding)


if __name__ == '__main__':
    unittest.main()
