#!/usr/bin/env python3
"""Cutover switch + clean revert (SPEC §7)."""
import unittest

from .config import load_pit_config
from .cutover import pe_dispatch_should_noop, plan_cutover, plan_from_flag


class TestCutover(unittest.TestCase):
    def test_live_cutover_disables_legacy_paths(self):
        cfg = load_pit_config({
            'pit': {
                'enabled': True, 'mode': 'live', 'cutover': True, 'max_boxes': 5,
            },
        })
        plan = plan_cutover(cfg)
        self.assertTrue(plan.cutover)
        self.assertFalse(plan.trainer_dequeue)
        self.assertFalse(plan.warm_pool)
        self.assertFalse(plan.standing_lanes)
        self.assertEqual(plan.trainer_fallback_hours, 24)
        self.assertEqual(plan.item_box_cap, 5)
        self.assertTrue(pe_dispatch_should_noop(cfg))

    def test_revert_restores_legacy_paths(self):
        cfg = load_pit_config({
            'pit': {
                'enabled': False, 'mode': 'live', 'revert': True,
            },
        })
        plan = plan_cutover(cfg)
        self.assertTrue(plan.revert)
        self.assertFalse(plan.cutover)
        self.assertTrue(plan.trainer_dequeue)
        self.assertTrue(plan.warm_pool)
        self.assertTrue(plan.standing_lanes)
        self.assertFalse(pe_dispatch_should_noop(cfg))

    def test_shadow_does_not_cutover(self):
        cfg = load_pit_config({'pit': {'enabled': True, 'mode': 'shadow'}})
        plan = plan_cutover(cfg)
        self.assertFalse(plan.cutover)
        # Shadow still owns dispatch when enabled.
        self.assertTrue(pe_dispatch_should_noop(cfg))

    def test_cutover_and_revert_mutex(self):
        with self.assertRaises(ValueError):
            load_pit_config({
                'pit': {'enabled': True, 'mode': 'live', 'cutover': True, 'revert': True},
            })

    def test_plan_from_flag(self):
        plan = plan_from_flag({'pit': {'enabled': True, 'mode': 'live', 'cutover': True}})
        self.assertTrue(plan.cutover)
        self.assertIn('trainer-dequeue-off', plan.notes)

    def test_live_refuses_per_round_root_mode(self):
        with self.assertRaises(ValueError):
            load_pit_config({
                'pit': {
                    'enabled': True, 'mode': 'live', 'cutover': True,
                    'max_boxes': 5, 'root_mode': 'per_round',
                },
            })

    def test_live_defaults_to_delegated_host(self):
        cfg = load_pit_config({
            'pit': {'enabled': True, 'mode': 'live', 'cutover': True, 'max_boxes': 5},
        })
        self.assertEqual(cfg.root_mode, 'delegated_host')


if __name__ == '__main__':
    unittest.main()
