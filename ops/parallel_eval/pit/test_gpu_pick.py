#!/usr/bin/env python3
"""Rent picker: B300-first-then-cheapest, ceiling, 8-GPU-only, RTX5090 config."""
from __future__ import annotations

import unittest

from ops.parallel_eval.pit import GPU_ORDER as PIT_ORDER, PRICE_CEILING_USD_H
from ops.parallel_eval.pit.config import load_pit_config
from ops.parallel_eval.pit.gpu_policy import (
    ALLOWED_GPUS,
    GPU_ORDER,
    pick_best_offer,
)
from ops.parallel_eval.pit.train_shard import (
    needs_optimizer_shard,
    numeric_equivalence_step,
)


def _row(huid: str, price: float, *, avail: int = 8, total: int = 8) -> dict:
    return {
        'huid': huid,
        'name': huid,
        'price_per_hour': price,
        'available_gpu_count': avail,
        'gpu_count': total,
    }


class TestPickBestOffer(unittest.TestCase):
    def test_b300_preferred_even_when_5090_cheaper(self):
        cands = [
            ('RTX5090', _row('rtx-a', 4.5), 4.5),
            ('B300', _row('b300-a', 40.0), 40.0),
            ('B300', _row('b300-b', 55.0), 55.0),
            ('H100', _row('h100-a', 20.0), 20.0),
        ]
        g, row = pick_best_offer(cands, order=GPU_ORDER, ceiling=56.0)
        self.assertEqual(g, 'B300')
        self.assertEqual(row['huid'], 'b300-a')  # cheapest B300 under ceiling

    def test_fallback_to_cheapest_allowed_when_no_b300(self):
        cands = [
            ('RTX5090', _row('rtx-a', 6.8), 6.8),
            ('RTX5090', _row('rtx-b', 4.0), 4.0),
            ('H200', _row('h200-a', 30.0), 30.0),
            ('H100', _row('h100-a', 18.0), 18.0),
        ]
        g, row = pick_best_offer(cands, order=GPU_ORDER, ceiling=56.0)
        self.assertEqual(g, 'RTX5090')
        self.assertEqual(row['huid'], 'rtx-b')

    def test_ceiling_filters_overpriced_b300(self):
        cands = [
            ('B300', _row('b300-hi', 80.0), 80.0),
            ('RTX5090', _row('rtx-a', 5.0), 5.0),
        ]
        g, row = pick_best_offer(cands, order=GPU_ORDER, ceiling=56.0)
        self.assertEqual(g, 'RTX5090')
        self.assertEqual(row['huid'], 'rtx-a')

    def test_mixed_offer_list_under_ceiling(self):
        cands = [
            ('B200', _row('b200-a', 45.0), 45.0),
            ('H200', _row('h200-a', 50.0), 50.0),
            ('H100', _row('h100-a', 22.0), 22.0),
            ('RTX5090', _row('rtx-a', 4.2), 4.2),
            ('B300', _row('b300-over', 70.0), 70.0),
        ]
        g, row = pick_best_offer(
            cands, order=('B300', 'RTX5090', 'B200', 'H200', 'H100'), ceiling=56.0,
        )
        # No B300 under ceiling → cheapest allowed = RTX5090 @ 4.2
        self.assertEqual(g, 'RTX5090')
        self.assertEqual(row['huid'], 'rtx-a')

    def test_nothing_under_ceiling(self):
        cands = [
            ('B300', _row('b300-hi', 90.0), 90.0),
            ('H100', _row('h100-hi', 60.0), 60.0),
        ]
        g, row = pick_best_offer(cands, order=GPU_ORDER, ceiling=56.0)
        self.assertIsNone(g)
        self.assertIsNone(row)

    def test_allowed_includes_rtx5090(self):
        self.assertIn('RTX5090', ALLOWED_GPUS)
        self.assertEqual(PIT_ORDER[0], 'B300')
        self.assertIn('RTX5090', PIT_ORDER)


class TestPitGpuConfig(unittest.TestCase):
    def test_accepts_rtx5090_in_gpu_order(self):
        cfg = load_pit_config({
            'pit': {
                'enabled': True, 'mode': 'live',
                'gpu_order': ['B300', 'RTX5090', 'B200', 'H200', 'H100'],
            },
        })
        self.assertEqual(cfg.gpu_order[1], 'RTX5090')
        self.assertEqual(cfg.price_ceiling_usd_h, PRICE_CEILING_USD_H)

    def test_default_order_is_b300_first(self):
        cfg = load_pit_config({'pit': {'enabled': True, 'mode': 'live'}})
        self.assertEqual(cfg.gpu_order[0], 'B300')
        self.assertIn('RTX5090', cfg.gpu_order)

    def test_unknown_gpu_refused(self):
        with self.assertRaises(ValueError):
            load_pit_config({
                'pit': {
                    'enabled': True, 'mode': 'live',
                    'gpu_order': ['B300', 'A100'],
                },
            })


class TestLowMemShard(unittest.TestCase):
    def test_needs_shard_under_100gib(self):
        self.assertTrue(needs_optimizer_shard(32.0))
        self.assertTrue(needs_optimizer_shard(80.0))
        self.assertFalse(needs_optimizer_shard(140.0))
        self.assertFalse(needs_optimizer_shard(None))

    def test_numeric_equivalence_cpu_offload(self):
        out = numeric_equivalence_step()
        self.assertLess(out['max_abs_diff'], 1e-5)
        self.assertLess(out['norm_diff'], 1e-5)


if __name__ == '__main__':
    unittest.main()
