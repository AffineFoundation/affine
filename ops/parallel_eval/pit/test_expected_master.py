#!/usr/bin/env python3
import unittest

from .expected_master import lookup_expected_master


H = 'a' * 64
M = 'b' * 64


class TestExpectedMaster(unittest.TestCase):
    def test_chain_override_wins(self):
        self.assertEqual(
            lookup_expected_master(H, chain_override={H: M}, public_manifest={H: 'c' * 64}),
            M,
        )

    def test_public_manifest(self):
        self.assertEqual(lookup_expected_master(H, public_manifest={'master_digest': M}), M)

    def test_index(self):
        self.assertEqual(lookup_expected_master(H, index={H: M}), M)

    def test_training_result(self):
        self.assertEqual(
            lookup_expected_master(H, training_result={'input_master_digest': M}),
            M,
        )

    def test_unknown_returns_none_not_empty(self):
        alerts = []
        out = lookup_expected_master(H, alert=lambda k, d: alerts.append(k))
        self.assertIsNone(out)
        self.assertNotEqual(out, '')
        self.assertEqual(alerts, ['expected-master-unpublished'])

    def test_bad_head(self):
        self.assertIsNone(lookup_expected_master('nope'))


if __name__ == '__main__':
    unittest.main()
