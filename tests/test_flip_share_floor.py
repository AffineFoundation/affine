"""Never-all-zero flip share floor."""
import unittest

from subnet.flip_share_floor import (apply_share_floor, gate_unknown_policy,
                                     positive_mass, normalize_shares)


class FlipShareFloorTests(unittest.TestCase):
    def test_candidate_with_mass_is_normalized(self):
        shares, meta = apply_share_floor({'a': 2.0, 'b': 2.0}, {'a': 1.0})
        self.assertFalse(meta['floor_applied'])
        self.assertTrue(meta['submit'])
        self.assertAlmostEqual(shares['a'], 0.5)
        self.assertAlmostEqual(shares['b'], 0.5)
        self.assertTrue(meta['never_burn'])

    def test_all_zero_candidate_retains_last_good(self):
        shares, meta = apply_share_floor({'a': 0.0, 'b': 0.0}, {'a': 3.0, 'b': 1.0})
        self.assertTrue(meta['floor_applied'])
        self.assertTrue(meta['submit'])
        self.assertEqual(meta['reason'], 'all-zero-candidate-retained-last-good')
        self.assertAlmostEqual(shares['a'], 0.75)
        self.assertAlmostEqual(shares['b'], 0.25)

    def test_no_last_good_refuses_submit(self):
        shares, meta = apply_share_floor({'a': 0.0}, None)
        self.assertFalse(meta['submit'])
        self.assertEqual(meta['reason'], 'no-last-good-refuse-all-zero')
        self.assertTrue(meta['never_burn'])

    def test_normalize_refuses_all_zero(self):
        self.assertTrue(positive_mass({'a': 0.1}))
        self.assertFalse(positive_mass({'a': 0.0}))
        with self.assertRaisesRegex(ValueError, 'all-zero'):
            normalize_shares({'a': 0.0})

    def test_unknown_policy_requires_floor_path(self):
        meta = gate_unknown_policy('strip_unverified', credit_requires_verify_pass=True)
        self.assertTrue(meta['share_floor_required'])
        self.assertTrue(meta['never_all_zero_weights'])
        with self.assertRaises(ValueError):
            gate_unknown_policy('burn-network', credit_requires_verify_pass=True)


if __name__ == '__main__':
    unittest.main()
