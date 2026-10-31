"""Prospective verifier admission must not silently replace miner generation."""
import copy
import tempfile
import types
import unittest
from unittest.mock import patch

from subnet import continuous_audit_policy as audit, symmetric_verifier_policy as policy
import test_symmetric_verifier_policy as fixtures

GENERATION = 'f5287f57bdae344e5a3d073309e5c830073f603838b4b52a189f5ec0a269cb24'


class StopBeforeDownload(Exception):
    pass


class CompatibilityTests(unittest.TestCase):
    def manifest(self):
        runtime, manifest = fixtures.CandidateTests().runtime()
        manifest.update(source_bundle={'sha256': GENERATION}, environment={'id': 'math'},
                        indices=[2], harness=runtime.harness, capabilities={'d' * 64: 'unused'})
        return manifest




    def test_historical_cohort_digest_is_unchanged(self):
        manifest = self.manifest()
        manifest.pop(policy.FIELD)
        manifest['numerical_policy'] = {'version': 'fixture'}
        fields = ('sampling_contract', 'sampling_source_hash', 'model_runtime_revision',
                  'backend_profile', 'numerical_policy', 'source_bundle')
        self.assertEqual(audit.verifier_contract(manifest), audit.digest({k: manifest[k] for k in fields}))

    def test_prospective_policy_has_distinct_cohort_without_changing_generation(self):
        manifest = self.manifest()
        manifest['numerical_policy'] = {'version': 'fixture'}
        old = copy.deepcopy(manifest)
        old.pop(policy.FIELD)
        self.assertNotEqual(audit.verifier_contract(old), audit.verifier_contract(manifest))
        changed = copy.deepcopy(manifest)
        changed[policy.FIELD]['source_files']['subnet/symmetric_verifier_policy.py'] = '0' * 64
        self.assertNotEqual(audit.verifier_contract(changed), audit.verifier_contract(manifest))
        self.assertEqual(old['source_bundle'], manifest['source_bundle'])


if __name__ == '__main__':
    unittest.main()
