"""Disposable signatures/maps only: these controls claim no GPU qualification."""
import base64
import copy
import unittest
from unittest.mock import patch

import test_retained_nll_horizon as horizon
from subnet import unaudited_training_execution as execution
from subnet.training_receipts import sha


class IntakeEquivalence(unittest.TestCase):
    def setUp(self):
        self.fixture = horizon.HorizonContract()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.sign = self.fixture.f.sign
        self.authority = self.fixture.f.authority
        self.release = copy.deepcopy(self.fixture.r)
        self.before = dict(self.release['execution_source_files'])
        for name in execution.INTAKE_FILES - {'subnet/training_task_representatives.py'}:
            self.before.setdefault(name, '1' * 64)
        for i in range(185 - len(self.before)):
            self.before[f'subnet/unchanged_fixture_{i}.py'] = sha({'fixture': i})
        self.assertEqual(len(self.before), 185)
        self.after = dict(self.before)
        self.after.update({name: sha({'new_intake': name}) for name in execution.INTAKE_FILES})
        self.addCleanup(patch.stopall)
        patch.object(execution, 'OBJECTIVE_INTAKE_BASE_MAP', sha(self.before)).start()

        original = dict(self.before)
        for name in execution.SCIENTIFIC_CHANGES | {'subnet/persistent_training_evidence.py',
                                                    'subnet/training_policy.py'}:
            original[name] = '0' * 64
        self.release.update(original_source_files=original, execution_source_files=self.after)
        oldq = copy.deepcopy(self.release['execution_qualification']['payload'])
        oldq.update(execution_source_bundle_sha256=execution.OBJECTIVE_INTAKE_BASE_BUNDLE,
                    execution_source_files_sha256=sha(self.before), retained_optimizer_step=1,
                    retained_input_checkpoint='7' * 64, retained_parent_descriptor_sha256='8' * 64)
        self.oldq = oldq
        self.old_document = self.sign(oldq)
        self.evidence = dict(
            version='retained-nll-native-intake-source-equivalence-evidence-v1',
            qualified_predecessor_qualification_sha256=sha(self.old_document),
            previous_source_files_sha256=sha(self.before), source_files_sha256=sha(self.after),
            changed_files={name: self.after[name] for name in execution.INTAKE_FILES},
            all_other_scientific_files_byte_identical=True, actual_GPU_execution=False,
            **{name: dict(sha256='d' * 64, passed=True) for name in (
                'installed_native_child', 'identical_input_emission', 'representative_integration_review')})
        self.qualification = dict(
            version=execution.OBJECTIVE_INTAKE_EQUIVALENCE, method=execution.OBJECTIVE_METHOD,
            execution_source_bundle_sha256=self.release['training_source_bundle']['sha256'],
            execution_source_files_sha256=sha(self.after), runtime_versions=oldq['runtime_versions'],
            training_runtime_sha256=oldq['training_runtime_sha256'], actual_GPU_execution=False,
            passed=True, optimizer_reset=False, objective_changed=False, hyperparameters_changed=False,
            effective_learning_rate=oldq['effective_learning_rate'],
            base_hyperparameters_sha256=oldq['base_hyperparameters_sha256'], state_version=oldq['state_version'],
            qualified_predecessor_qualification=self.old_document,
            qualified_predecessor_source_files=self.before, equivalence_evidence=self.sign(self.evidence))
        self.refresh()

    def refresh(self):
        self.release['changed_source_files'] = {
            name: value for name, value in self.after.items()
            if self.release['original_source_files'].get(name) != value}
        self.qualification['execution_source_files_sha256'] = sha(self.after)
        self.evidence['source_files_sha256'] = sha(self.after)
        self.evidence['changed_files'] = {
            name: value for name, value in self.after.items() if self.before.get(name) != value}
        self.qualification['equivalence_evidence'] = self.sign(self.evidence)

    def check(self):
        self.release['execution_qualification'] = self.sign(self.qualification)
        return execution.release(self.sign(self.release), self.authority)

    def test_full_release_distinguishes_old_gpu_from_new_cpu_evidence(self):
        result = self.check()
        q = result['execution_qualification']['payload']
        self.assertFalse(q['actual_GPU_execution'])
        self.assertTrue(q['qualified_predecessor_qualification']['payload']['actual_GPU_execution'])
        self.assertEqual(result['training_horizon']['first_optimizer_step'], 2)
        self.assertEqual(q['qualified_predecessor_qualification']['payload']['retained_optimizer_step'], 1)

    def test_scientific_change_rejected_even_with_resigned_consistent_metadata(self):
        self.after['subnet/task_normalized_training.py'] = 'f' * 64
        self.refresh()
        with self.assertRaises(ValueError):
            self.check()

    def test_missing_or_extra_source_members_rejected(self):
        self.after['subnet/unqualified.py'] = 'e' * 64
        self.refresh()
        with self.assertRaises(ValueError):
            self.check()

    def test_cannot_claim_combined_bundle_executed_on_gpu(self):
        self.qualification['actual_GPU_execution'] = True
        with self.assertRaises(ValueError):
            self.check()

    def test_actual_predecessor_gpu_evidence_still_required(self):
        self.oldq['actual_GPU_execution'] = False
        old = self.sign(self.oldq)
        self.qualification['qualified_predecessor_qualification'] = old
        self.evidence['qualified_predecessor_qualification_sha256'] = sha(old)
        self.refresh()
        with self.assertRaises(ValueError):
            self.check()

    def test_private_qualification_cannot_claim_live_moment_equivalence(self):
        self.oldq['live_optimizer_distribution_equivalence_claimed'] = True
        old = self.sign(self.oldq)
        self.qualification['qualified_predecessor_qualification'] = old
        self.evidence['qualified_predecessor_qualification_sha256'] = sha(old)
        self.refresh()
        with self.assertRaises(ValueError):
            self.check()

    def test_nested_signatures_are_checked(self):
        for field in ('qualified_predecessor_qualification', 'equivalence_evidence'):
            with self.subTest(field=field):
                old = copy.deepcopy(self.qualification[field])
                self.qualification[field]['signature'] = base64.b64encode(bytes(64)).decode()
                with self.assertRaises(Exception):
                    self.check()
                self.qualification[field] = old

    def test_exact_predecessor_archive_required(self):
        self.oldq['execution_source_bundle_sha256'] = 'f' * 64
        old = self.sign(self.oldq)
        self.qualification['qualified_predecessor_qualification'] = old
        self.evidence['qualified_predecessor_qualification_sha256'] = sha(old)
        self.refresh()
        with self.assertRaises(ValueError):
            self.check()

    def test_failed_cpu_equivalence_cannot_authorize_release(self):
        self.evidence['installed_native_child']['passed'] = False
        self.refresh()
        with self.assertRaises(ValueError):
            self.check()

    def test_evidence_cannot_bind_a_different_predecessor(self):
        self.evidence['qualified_predecessor_qualification_sha256'] = 'e' * 64
        self.refresh()
        with self.assertRaises(ValueError):
            self.check()

    def test_no_relative_objective_or_optimizer_change(self):
        for field in ('optimizer_reset', 'objective_changed', 'hyperparameters_changed'):
            with self.subTest(field=field):
                self.qualification[field] = True
                with self.assertRaises(ValueError):
                    self.check()
                self.qualification[field] = False

    def test_horizon_cannot_be_extended_by_new_intake_qualification(self):
        self.release['training_horizon']['updates'] = 17
        with self.assertRaises(ValueError):
            self.check()


if __name__ == '__main__':
    unittest.main()
