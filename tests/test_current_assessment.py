import unittest
from subnet.current_assessment import calculate, fallback, recipients


class AssessmentTests(unittest.TestCase):
    def snap(self, epoch, at, count=1, probability=.5, penalty=1, coverage=1):
        return dict(epoch=epoch, round=int(at // 3600), cutoff=36000,
                    miners={'m': dict(unique_eligible_batches=count, validity_probability=probability,
                                      reward_multiplier=penalty, resolution_coverage_factor=coverage)})

    def test_six_hour_decay(self):
        a = calculate([self.snap('a', 14400)], {'a': 14400}, 36000)
        b = self.snap('a', 14400); b['cutoff'] = 14400
        start = calculate([b], {'a': 14400}, 14400)
        self.assertAlmostEqual(a['points']['m'], start['points']['m'] / 2)

    def test_training_failure_and_missing_opening_are_not_inputs(self):
        result = calculate([self.snap('failed-training', 36000, count=3)], {'failed-training': 36000}, 36000)
        self.assertGreater(result['points']['m'], 0)
        self.assertFalse(result['training_completion_required'])

    def test_immediate_penalty_not_smoothed(self):
        a = calculate([self.snap('a', 36000, count=3)], {'a': 36000}, 36000)
        b = calculate([self.snap('a', 36000, count=3, penalty=.1)], {'a': 36000}, 36000)
        self.assertAlmostEqual(b['points']['m'], a['points']['m'] * .1)

    def test_zero_duplicate_counts_and_unknown_coverage(self):
        a = calculate([self.snap('a', 36000, count=0)], {'a': 36000}, 36000)
        self.assertEqual(a['points']['m'], 0)
        b = calculate([self.snap('a', 36000, coverage=0)], {'a': 36000}, 36000)
        self.assertEqual(b['points']['m'], 0)

    def test_latest_estimate_reassesses_history_without_recount(self):
        a = self.snap('a', 32400, count=2)
        b = self.snap('b', 36000, count=0, probability=.1)
        result = calculate([a, b], {'a': 32400, 'b': 36000}, 36000)
        a['miners']['m']['validity_probability'] = .1
        baseline = calculate([a], {'a': 32400}, 36000)
        self.assertEqual(result['points'], baseline['points'])

    def test_outage_keeps_assessment_inactivity_decays(self):
        a = calculate([self.snap('a', 36000)], {'a': 36000}, 36000)
        self.assertEqual(fallback(a, 39600, 'TimeoutError')['points'], a['points'])
        self.assertTrue(fallback(a, 39600, 'TimeoutError')['assessment_stale'])

    def test_mapping_excludes_departed_identity_only(self):
        a = {'points': {'m': 1, 'departed': 1}}
        p, r, x = recipients(a, {'hotkey': {'public_key': 'm', 'uid': 85}})
        self.assertEqual(set(p), {'hotkey'}); self.assertEqual(x, ['departed'])
        self.assertEqual(r['hotkey']['uid'], 85)

    def test_nan_future_duplicate_reject(self):
        for snapshots, times in (([self.snap('a', 36000, probability=float('nan'))], {'a':36000}),
                                  ([self.snap('a', 36000)], {'a':36001}),
                                  ([self.snap('a', 36000)]*2, {'a':36000})):
            with self.assertRaises(ValueError):calculate(snapshots,times,36000)

if __name__ == '__main__':unittest.main()
