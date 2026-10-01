"""Authenticated dataset identity controls; no native or model execution."""
import copy
import importlib
import unittest
from subnet import native_tau2_common_bridge_seeded as seeded
from subnet.native_tau2_common_search_contract import digest

class SeededHeldoutTests(unittest.TestCase):
    def setUp(self):
        self.fixture = importlib.import_module('test_native_tau2_common_bridge').TestCommonBridge()
        self.fixture.setUp()
    def contract(self):
        f = self.fixture
        return seeded.heldout_contract(f.f.sign(f.f.manifest), f.f.authority, f.f.user, f.public)
    def test_changed_agent_seed_changes_dataset(self):
        before = self.contract()
        self.fixture.f.manifest['roles']['agent']['seed_start'] += 1
        after = self.contract()
        self.assertNotEqual(before['dataset_id'], after['dataset_id'])
        self.assertEqual(after['agent_geometry_and_policy']['seed_start'], 51)
        self.assertEqual(after['version'], seeded.VERSION)
    def test_weight_only_successor_preserves_dataset(self):
        before = self.contract()
        m = self.fixture.f.manifest
        cp = m['roles']['agent']['checkpoint']
        cp['files']['model.safetensors'] = 'f' * 64
        cp['id'] = digest(cp['files'])
        m['checkpoint'] = copy.deepcopy(cp)
        self.assertEqual(before['dataset_id'], self.contract()['dataset_id'])
    def test_wrong_seed_type_or_policy_fails_authentication_contract(self):
        for field, value in [('seed_start', True), ('seed_policy', 'unapproved')]:
            self.setUp()
            self.fixture.f.manifest['roles']['agent'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.contract()
    def test_fixed_auxiliary_seed_still_affects_dataset(self):
        before = self.contract()
        self.fixture.f.user['seed_start'] += 1
        self.fixture.f.manifest['roles']['user'] = copy.deepcopy(self.fixture.f.user)
        self.assertNotEqual(before['dataset_id'], self.contract()['dataset_id'])
    def test_prospective_streaming_uses_seeded_contract(self):
        from subnet import native_tau2_common_service_streaming as streaming
        self.assertIs(streaming.heldout_contract, seeded.heldout_contract)
