"""Portable signature/mutation controls for the narrow logging qualification.

Synthetic release inventories use an ephemeral test authority. Only the two
inventory digests are patched to those fixtures; deployed module hashes and all
qualification guards remain the actual implementation. Existing release suites
exercise historical release validation separately.
"""
import base64
import copy
import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch
from nacl.signing import SigningKey
from subnet import unaudited_training_execution as m
from subnet.training_receipts import authenticate

ROOT = Path(__file__).resolve().parents[1]

class LoggingAdmission(unittest.TestCase):
    def setUp(self):
        self.key = SigningKey.generate()
        self.authority = self.key.verify_key.encode().hex()
        before = {'subnet/fixture_%03d.py' % i: '%064x' % (i+1) for i in range(185)}
        before.update({'subnet/task_normalized_training.py': '4ba6373cbbb15e996603e57108b833a42e4b4e3d79669a68be14773fc8856421',
                       'subnet/unaudited_training_execution.py': m.LOGGING_ORIGINAL_PARSER_SHA})
        self.before = before
        self.after = dict(before)
        for name in m.LOGGING_FILES:
            self.after[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        qualified = dict(self.after)
        qualified['subnet/unaudited_training_execution.py'] = m.LOGGING_ORIGINAL_PARSER_SHA
        for name, value in [('LOGGING_BASE_MAP', m.sha(before)), ('LOGGING_QUALIFIED_MAP', m.sha(qualified))]:
            p = patch.object(m, name, value); p.start(); self.addCleanup(p.stop)
        self.previous = dict(version=m.THREAD_RELEASE, execution_source_files=before,
            execution_qualification=self.sign(dict(version=m.THREAD_QUALIFICATION)),
            method='fixture-method', runtime_versions={'torch': 'fixture'},
            training_runtime_sha256='a'*64, training_policy='fixture-policy',
            training_input_policy='fixture-input', effective_learning_rate=5e-7,
            original_source_files=before, original_source_bundle_sha256='b'*64,
            training_horizon={'fixture': True}, steps=1, optimizer_threading={'fixture': 16},
            genesis_sha256='c'*64, hyperparameters_sha256='d'*64)
        self.original = self.sign(self.previous)
        def previous_only(document, authority):
            self.assertEqual(document, self.original)
            self.assertEqual(authority, self.authority)
            return authenticate(document, authority)
        p = patch.object(m, 'release', side_effect=previous_only); p.start(); self.addCleanup(p.stop)

    def sign(self, payload):
        from subnet.training_receipts import canonical
        return dict(payload=payload, signer=self.authority,
                    signature=base64.b64encode(self.key.sign(canonical(payload)).signature).decode())

    def fixture(self):
        value = copy.deepcopy(self.previous)
        value['execution_source_files'] = self.after
        value['training_source_bundle'] = {'sha256': 'e'*64}
        changed = {n: h for n, h in self.after.items() if self.before.get(n) != h}
        evidence = dict(version='detached-sequence-component-logging-evidence-v1',
            predecessor_release_sha256=m.sha(self.original), baseline_files_sha256=m.LOGGING_BASE_MAP,
            qualified_files_sha256=m.LOGGING_QUALIFIED_MAP, execution_files_sha256=m.sha(self.after),
            changed_files=changed, H200_review_sha256=m.LOGGING_H200_REVIEW_SHA,
            original_result_sha256='1'*64, durable_capture_sha256='2'*64, cpu_controls_sha256='3'*64,
            runtime_versions=value['runtime_versions'], training_runtime_sha256=value['training_runtime_sha256'])
        for name in ('actual_private_H200_execution', 'all_gradients_master_moments_BF16_equal',
                     'same_forward_counts', 'scalar_diagnostics_passed', 'overhead_reviewed',
                     'original_capture_complete', 'passed'):
            evidence[name] = True
        for name in ('optimizer_reset', 'objective_changed', 'hyperparameters_changed', 'new_production_state_qualification_claimed'):
            evidence[name] = False
        q = dict(version=m.LOGGING_QUALIFICATION, qualified_predecessor_release=self.original,
            execution_source_bundle_sha256='e'*64, execution_source_files_sha256=m.sha(self.after),
            runtime_versions=value['runtime_versions'], training_runtime_sha256=value['training_runtime_sha256'],
            qualification_evidence=self.sign(evidence))
        value['execution_qualification'] = self.sign(q)
        return value, q, evidence

    def test_exact_deployed_trainer_and_helper_hashes(self):
        self.assertEqual(self.after['subnet/task_normalized_training.py'], m.LOGGING_TRAINER_SHA)
        self.assertEqual(self.after['subnet/sequence_component_diagnostics.py'], m.LOGGING_HELPER_SHA)

    def test_signed_additive_qualification_accepted(self):
        value, _, _ = self.fixture()
        self.assertIsNone(m._qualification(value, self.authority))

    def test_missing_or_bad_evidence_rejected(self):
        for key in ('original_capture_complete', 'passed', 'all_gradients_master_moments_BF16_equal', 'same_forward_counts'):
            value, q, e = self.fixture(); e[key] = False
            q['qualification_evidence'] = self.sign(e); value['execution_qualification'] = self.sign(q)
            with self.subTest(key=key), self.assertRaises(ValueError): m._qualification(value, self.authority)

    def test_forged_qualification_signature_rejected(self):
        value, _, _ = self.fixture(); value['execution_qualification']['signature'] = base64.b64encode(bytes(64)).decode()
        with self.assertRaises(Exception): m._qualification(value, self.authority)

    def test_scientific_or_lineage_changes_rejected(self):
        for key, replacement in [('steps', 2), ('effective_learning_rate', 1e-6), ('genesis_sha256', '0'*64), ('hyperparameters_sha256', '0'*64)]:
            value, _, _ = self.fixture(); value[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError): m._qualification(value, self.authority)

    def test_unqualified_source_changes_rejected(self):
        value, q, _ = self.fixture(); value['execution_source_files'] = dict(self.after)
        value['execution_source_files']['subnet/task_normalized_training.py'] = '0'*64
        q['execution_source_files_sha256'] = m.sha(value['execution_source_files'])
        value['execution_qualification'] = self.sign(q)
        with self.assertRaises(ValueError): m._qualification(value, self.authority)

    def test_recursive_logging_predecessor_rejected(self):
        value, q, _ = self.fixture(); previous = copy.deepcopy(self.previous)
        previous['execution_qualification'] = self.sign({'version': m.LOGGING_QUALIFICATION})
        q['qualified_predecessor_release'] = self.sign(previous); value['execution_qualification'] = self.sign(q)
        with self.assertRaisesRegex(ValueError, 'non-logging'): m._qualification(value, self.authority)

    def test_new_optimizer_state_qualification_claim_rejected(self):
        value, q, e = self.fixture(); e['new_production_state_qualification_claimed'] = True
        q['qualification_evidence'] = self.sign(e); value['execution_qualification'] = self.sign(q)
        with self.assertRaises(ValueError): m._qualification(value, self.authority)

if __name__ == '__main__': unittest.main()
