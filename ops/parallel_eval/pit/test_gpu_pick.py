#!/usr/bin/env python3
"""Rent picker: B300-first-then-cheapest, ceiling, 8-GPU-only, sharded-GPU gate."""
from __future__ import annotations

import unittest
from unittest import mock

from ops.parallel_eval.pit import (
    GPU_ORDER as PIT_ORDER,
    PRICE_CEILING_USD_H,
    SHARDED_GPUS,
)
from ops.parallel_eval.pit.config import load_pit_config
from ops.parallel_eval.pit.gpu_policy import (
    ALLOWED_GPUS,
    GPU_ORDER,
    effective_gpu_order,
    pick_best_offer,
)
from ops.parallel_eval.pit.train_shard import (
    needs_optimizer_shard,
    numeric_equivalence_step,
    numeric_equivalence_zero3,
    preflight_or_raise,
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
        g, row = pick_best_offer(
            cands, order=GPU_ORDER, ceiling=56.0, allow_sharded_gpus=True,
        )
        self.assertEqual(g, 'B300')
        self.assertEqual(row['huid'], 'b300-a')

    def test_fallback_to_cheapest_allowed_when_no_b300(self):
        cands = [
            ('RTX5090', _row('rtx-a', 6.8), 6.8),
            ('RTX5090', _row('rtx-b', 4.0), 4.0),
            ('H200', _row('h200-a', 30.0), 30.0),
            ('H100', _row('h100-a', 18.0), 18.0),
        ]
        g, row = pick_best_offer(
            cands, order=GPU_ORDER, ceiling=56.0, allow_sharded_gpus=True,
        )
        # No B300 → cheapest under ceiling among order = RTX5090 @ 4.0
        # (order is B300,B200,H200,RTX5090,H100; preferred B300 missing → global cheapest)
        self.assertEqual(g, 'RTX5090')
        self.assertEqual(row['huid'], 'rtx-b')

    def test_gate_blocks_5090_by_default(self):
        cands = [
            ('B300', _row('b300-hi', 80.0), 80.0),
            ('RTX5090', _row('rtx-a', 5.0), 5.0),
            ('H200', _row('h200-a', 30.0), 30.0),
        ]
        g, row = pick_best_offer(
            cands, order=GPU_ORDER, ceiling=56.0, allow_sharded_gpus=False,
        )
        self.assertEqual(g, 'H200')
        self.assertEqual(row['huid'], 'h200-a')

    def test_ceiling_filters_overpriced_b300(self):
        cands = [
            ('B300', _row('b300-hi', 80.0), 80.0),
            ('B200', _row('b200-a', 45.0), 45.0),
            ('RTX5090', _row('rtx-a', 5.0), 5.0),
        ]
        g, row = pick_best_offer(
            cands, order=GPU_ORDER, ceiling=56.0, allow_sharded_gpus=False,
        )
        self.assertEqual(g, 'B200')

    def test_mixed_offer_list_under_ceiling(self):
        cands = [
            ('B200', _row('b200-a', 45.0), 45.0),
            ('H200', _row('h200-a', 50.0), 50.0),
            ('H100', _row('h100-a', 22.0), 22.0),
            ('RTX5090', _row('rtx-a', 4.2), 4.2),
            ('B300', _row('b300-over', 70.0), 70.0),
        ]
        g, row = pick_best_offer(
            cands, order=GPU_ORDER, ceiling=56.0, allow_sharded_gpus=True,
        )
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
        self.assertEqual(PIT_ORDER, ('B300', 'B200', 'H200', 'RTX5090', 'H100'))
        self.assertIn('RTX5090', SHARDED_GPUS)


class TestPitGpuConfig(unittest.TestCase):
    def test_accepts_rtx5090_in_gpu_order(self):
        cfg = load_pit_config({
            'pit': {
                'enabled': True, 'mode': 'live',
                'gpu_order': ['B300', 'B200', 'H200', 'RTX5090', 'H100'],
                'allow_sharded_gpus': True,
            },
        })
        self.assertEqual(cfg.gpu_order[-2], 'RTX5090')
        self.assertEqual(cfg.price_ceiling_usd_h, PRICE_CEILING_USD_H)
        self.assertEqual(cfg.effective_gpu_order()[-2], 'RTX5090')

    def test_default_order_gates_sharded(self):
        cfg = load_pit_config({'pit': {'enabled': True, 'mode': 'live'}})
        self.assertEqual(cfg.gpu_order[0], 'B300')
        self.assertIn('RTX5090', cfg.gpu_order)
        self.assertFalse(cfg.allow_sharded_gpus)
        self.assertEqual(cfg.effective_gpu_order(), ('B300', 'B200', 'H200'))
        self.assertEqual(effective_gpu_order(GPU_ORDER), ('B300', 'B200', 'H200'))

    def test_unknown_gpu_refused(self):
        with self.assertRaises(ValueError):
            load_pit_config({
                'pit': {
                    'enabled': True, 'mode': 'live',
                    'gpu_order': ['B300', 'A100'],
                },
            })

    def test_effective_order_fail_closed_when_all_sharded(self):
        with self.assertRaises(ValueError):
            effective_gpu_order(
                ('RTX5090', 'H100'), allow_sharded_gpus=False,
            )


class TestPeCommonGpuOrderFailClosed(unittest.TestCase):
    def _load_pe_common(self):
        import importlib.util
        import sys
        import types
        from pathlib import Path
        repo = Path(__file__).resolve().parents[3]
        live = str(repo / 'ops' / 'parallel_eval' / 'live')
        if live not in sys.path:
            sys.path.insert(0, live)
        # pe_common imports replace_pod at module load — stub it.
        rp = types.ModuleType('replace_pod')
        rp.Pod = type('Pod', (), {})
        path = repo / 'ops' / 'parallel_eval' / 'live' / 'pe_common.py'
        with mock.patch.dict(sys.modules, {'replace_pod': rp}):
            # Drop any prior stub load so we re-exec with the mock present.
            sys.modules.pop('pe_common_under_test', None)
            spec = importlib.util.spec_from_file_location('pe_common_under_test', path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

    def test_gpu_order_fail_closed_when_filter_empties(self):
        PC = self._load_pe_common()
        with mock.patch.object(PC, 'pe_cfg', return_value={
            'pit': {'gpu_order': ['RTX5090', 'H100'], 'allow_sharded_gpus': False},
        }), mock.patch.object(PC, 'allow_sharded_gpus', return_value=False):
            with self.assertRaises(ValueError):
                PC.gpu_order()

    def test_pick_best_offer_fail_closed_when_filter_empties(self):
        PC = self._load_pe_common()
        cands = [('RTX5090', _row('rtx', 5.0), 5.0)]
        with mock.patch.object(PC, 'allow_sharded_gpus', return_value=False):
            with self.assertRaises(ValueError):
                PC.pick_best_offer(cands, order=('RTX5090', 'H100'), ceiling=56.0)


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

    def test_numeric_equivalence_zero3_tiny(self):
        out = numeric_equivalence_zero3(seed=7, world=8)
        self.assertTrue(out['ok'], out)
        self.assertLessEqual(out['max_abs_diff'], out['tol'])

    def test_preflight_refuses_when_over_vram(self):
        # Unsharded 9B-scale params on 32 GiB must refuse.
        n_params = int(9.0e9)
        with self.assertRaises(Exception) as ctx:
            preflight_or_raise(
                n_params=n_params, per_gpu_gib=32.0, world=1, sharded=False,
            )
        self.assertIn('exceeds VRAM', str(ctx.exception))

    def test_preflight_accepts_sharded_8x32(self):
        n_params = int(9.0e9)
        out = preflight_or_raise(
            n_params=n_params, per_gpu_gib=32.0, world=8, sharded=True,
            token_budget=24576,
        )
        self.assertLessEqual(out['peak_gib'], 32.0)


if __name__ == '__main__':
    unittest.main()
