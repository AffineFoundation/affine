import dataclasses
import math
import unittest

from .guard import Thresholds, Trajectory, RatioInputError, assess


def row(deltas, *, task='task', nonce=0, advantage=1., outcome='correct', active=True):
    return Trajectory(task, nonce, outcome, advantage,
                      tuple(-10. for _ in deltas), tuple(-10.+d for d in deltas), active)


class GuardTests(unittest.TestCase):
    def test_identity_and_inputs_unchanged(self):
        r = row([0.]*20)
        original = dataclasses.asdict(r)
        result = assess([r])
        self.assertTrue(result['passed'])
        self.assertEqual(dataclasses.asdict(r), original)
        self.assertFalse(result['proof_acceptance'])

    def test_sparse_tail_passes_without_demanding_single_token_ppo_match(self):
        report = assess([row([.26]+[0.]*999)])
        self.assertTrue(report['passed'])
        self.assertEqual(report['aggregate']['outside_ppo_band_tokens'], 1)

    def test_one_extreme_token_fails_even_if_pooled_statistics_small(self):
        report = assess([row([1.5]+[0.]*9999)])
        self.assertFalse(report['passed'])
        self.assertIn('max_abs_logratio', [r['metric'] for r in report['failures']])

    def test_finite_huge_difference_fails_without_exponent_overflow(self):
        r = Trajectory('t', 0, 'correct', 1., (-1000.,), (-1.,))
        self.assertFalse(assess([r])['passed'])

    def test_systematic_small_bias_fails_mean(self):
        report = assess([row([.021]*1000)])
        self.assertIn('mean_abs_logratio', [r['metric'] for r in report['failures']])

    def test_p99_tail_independent_of_mean_and_clip(self):
        report = assess([row([.16]*20+[0.]*980)])
        self.assertEqual([r['metric'] for r in report['failures']], ['p99_abs_logratio'])

    def test_pooled_fraction_fails_independently(self):
        limits = Thresholds(mean_abs_logratio=.1, p99_abs_logratio=.5)
        report = assess([row([.25]*11+[0.]*989)], limits)
        self.assertEqual([r['metric'] for r in report['failures']], ['pooled_ppo_clip_fraction'])

    def test_short_poison_trajectory_cannot_hide_in_long_clean_rows(self):
        report = assess([row([.25]*10), row([0.]*10000, nonce=1)])
        self.assertEqual([r['metric'] for r in report['failures']], ['max_per_trajectory_clip_fraction'])

    def test_sign_aware_clipping_distinguishes_sides(self):
        limits = Thresholds(1., 1., 1., 1., 1.)
        positive = row([.3, -.3], nonce=0)
        negative = row([.3, -.3], nonce=1, advantage=-1., outcome='wrong')
        report = assess([positive, negative], limits)
        self.assertEqual(report['aggregate']['outside_ppo_band_tokens'], 4)
        self.assertEqual(report['aggregate']['sign_aware_clipped_tokens'], 2)
        self.assertEqual(report['aggregate']['sign_aware_clipped_fraction'], .5)

    def test_zero_advantage_has_no_false_zero_clipping_rate(self):
        report = assess([row([0.]*10, advantage=0.)])
        self.assertIsNone(report['aggregate']['sign_aware_clipped_fraction'])

    def test_inactive_unresolved_cannot_dilute_training_population(self):
        report = assess([row([.25]*10), row([0.]*10000, nonce=1,
                                          active=False, outcome='unresolved')])
        self.assertFalse(report['passed'])
        self.assertEqual(report['aggregate']['tokens'], 10)
        with self.assertRaises(RatioInputError):
            assess([row([0.], outcome='unresolved')])

    def test_exact_inclusive_aggregate_boundaries(self):
        r = row([.25]+[0.]*99)
        broad = assess([r], Thresholds(1., 1., 1., 1., 1.))['aggregate']
        limits = Thresholds(**{k:broad[k] for k in ['max_abs_logratio','mean_abs_logratio',
            'p99_abs_logratio','pooled_ppo_clip_fraction','max_per_trajectory_clip_fraction']})
        self.assertTrue(assess([r], limits)['passed'])
        for field in ['max_abs_logratio','mean_abs_logratio','p99_abs_logratio',
                      'pooled_ppo_clip_fraction','max_per_trajectory_clip_fraction']:
            narrowed = dataclasses.replace(limits, **{field:math.nextafter(getattr(limits, field), -math.inf)})
            self.assertFalse(assess([r], narrowed)['passed'], field)

    def test_alignment_empty_duplicate_and_bad_types(self):
        malformed = [dataclasses.replace(row([0.]), current_logprobs=()),
                     dataclasses.replace(row([0.]), behavior_logprobs=(True,)),
                     dataclasses.replace(row([0.]), nonce=True),
                     dataclasses.replace(row([0.]), active=1),
                     dataclasses.replace(row([0.]), outcome='failure'),
                     dataclasses.replace(row([0.]), advantage=True)]
        for r in malformed:
            with self.subTest(r=r), self.assertRaises(RatioInputError): assess([r])
        with self.assertRaises(RatioInputError): assess([])
        with self.assertRaises(RatioInputError): assess([row([0.]), row([0.])])
        with self.assertRaises(RatioInputError): assess([row([0.], active=False)])

    def test_nonfinite_and_positive_logprob_never_pass(self):
        for value in [float('nan'), float('inf'), -float('inf'), .01]:
            for field in ['behavior_logprobs', 'current_logprobs']:
                with self.subTest(value=value, field=field), self.assertRaises(RatioInputError):
                    assess([dataclasses.replace(row([0.]), **{field:(value,)})])

    def test_invalid_thresholds(self):
        for changes in [{'max_abs_logratio':True}, {'mean_abs_logratio':float('nan')},
                        {'pooled_ppo_clip_fraction':1.1}, {'clip_epsilon':0.},
                        {'clip_epsilon':1.}, {'p99_abs_logratio':-1.}]:
            with self.subTest(changes=changes), self.assertRaises(RatioInputError):
                assess([row([0.])], dataclasses.replace(Thresholds(), **changes))


if __name__ == '__main__':
    unittest.main()
