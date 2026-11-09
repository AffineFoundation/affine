#!/usr/bin/env python3
"""E1041 noise-tolerant shadow pass criterion (not equal hashes / equal z)."""
import unittest

from .shadow_compare import (
    DISAGREE_RATIO_HI,
    DISAGREE_RATIO_LO,
    Z_GAP_MAX,
    evaluate_shadow_pass,
)


def _report(**overrides):
    base = {
        'source_file_hashes': {'trainer': 'a' * 64, 'batch': 'b' * 64},
        'batching': {'tasks': 125, 'rollouts': 1000},
        'tokens': {'prompt_tokens': 100, 'output_tokens': 200, 'loss_tokens': 150},
        'update_size': 1.0,
        'gradient_norm': 1.0,
    }
    base.update(overrides)
    return base


def _meta(**overrides):
    base = {
        'round_id': 'r1',
        'plan_seed': 'seed-1',
        'heldout': {'round_id': 'r1', 'settings_hash': 's' * 64},
    }
    base.update(overrides)
    return base


def _per_task(box_wins=47, live_wins=50, shared=97):
    """Build per_task maps whose disagreement ratio ≈ box_wins/live_wins."""
    box, live = {}, {}
    for i in range(shared):
        key = 't%d' % i
        if i < box_wins:
            box[key] = {'c': 2, 'n': 2}
            live[key] = {'c': 1, 'n': 2}
        elif i < box_wins + live_wins:
            box[key] = {'c': 1, 'n': 2}
            live[key] = {'c': 2, 'n': 2}
        else:
            box[key] = {'c': 1, 'n': 2}
            live[key] = {'c': 1, 'n': 2}
    return box, live


class TestShadowCompare(unittest.TestCase):
    def test_pass_within_noise(self):
        box_pt, live_pt = _per_task(47, 50, 97)  # ratio 0.94
        out = evaluate_shadow_pass(
            box_report=_report(update_size=1.0005, gradient_norm=1.02),
            live_report=_report(update_size=1.0, gradient_norm=1.0),
            box_meta=_meta(),
            live_meta=_meta(),
            box_per_task=box_pt,
            live_per_task=live_pt,
            box_z=-0.48,
            live_z=0.64,
            train_gpu='B300',
            eval_gpu='B300',
            live_train_gpu='H200',
            live_eval_gpu='H200',
        )
        self.assertTrue(out['passed'])
        self.assertFalse(out['require_equal_candidate_hashes'])
        self.assertFalse(out['require_equal_z'])
        self.assertEqual(out['train_gpu'], 'B300')
        self.assertEqual(out['live_train_gpu'], 'H200')
        self.assertGreaterEqual(out['disagreement_ratio'], DISAGREE_RATIO_LO)
        self.assertLessEqual(out['disagreement_ratio'], DISAGREE_RATIO_HI)
        self.assertLess(out['checks']['z_gap']['gap'], Z_GAP_MAX)

    def test_fail_on_hash_mismatch(self):
        out = evaluate_shadow_pass(
            box_report=_report(source_file_hashes={'batch': '1' * 64}),
            live_report=_report(source_file_hashes={'batch': '2' * 64}),
            box_meta=_meta(),
            live_meta=_meta(),
            box_per_task=_per_task()[0],
            live_per_task=_per_task()[1],
            box_z=0.1,
            live_z=0.2,
        )
        self.assertFalse(out['passed'])
        self.assertFalse(out['checks']['input_hashes']['ok'])

    def test_fail_on_update_size(self):
        out = evaluate_shadow_pass(
            box_report=_report(update_size=1.01),  # > 1e-3 rel
            live_report=_report(update_size=1.0),
            box_meta=_meta(),
            live_meta=_meta(),
            box_per_task=_per_task()[0],
            live_per_task=_per_task()[1],
            box_z=0.1,
            live_z=0.2,
        )
        self.assertFalse(out['passed'])
        self.assertFalse(out['checks']['update_size']['ok'])

    def test_fail_on_grad_norm(self):
        out = evaluate_shadow_pass(
            box_report=_report(gradient_norm=1.10),  # > 5%
            live_report=_report(gradient_norm=1.0),
            box_meta=_meta(),
            live_meta=_meta(),
            box_per_task=_per_task()[0],
            live_per_task=_per_task()[1],
            box_z=0.1,
            live_z=0.2,
        )
        self.assertFalse(out['passed'])
        self.assertFalse(out['checks']['gradient_norm']['ok'])

    def test_fail_on_z_gap(self):
        out = evaluate_shadow_pass(
            box_report=_report(),
            live_report=_report(),
            box_meta=_meta(),
            live_meta=_meta(),
            box_per_task=_per_task()[0],
            live_per_task=_per_task()[1],
            box_z=0.0,
            live_z=2.5,
        )
        self.assertFalse(out['passed'])
        self.assertFalse(out['checks']['z_gap']['ok'])

    def test_fail_on_draw_mismatch(self):
        out = evaluate_shadow_pass(
            box_report=_report(),
            live_report=_report(),
            box_meta=_meta(plan_seed='seed-a'),
            live_meta=_meta(plan_seed='seed-b'),
            box_per_task=_per_task()[0],
            live_per_task=_per_task()[1],
            box_z=0.1,
            live_z=0.2,
        )
        self.assertFalse(out['passed'])
        self.assertFalse(out['checks']['draw_seed_baseline']['ok'])


if __name__ == '__main__':
    unittest.main()
