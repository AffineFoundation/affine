"""Signed v5/v6 continuation reports retain their authorized learning rate."""
import copy
import unittest

import test_authorized_lr_continuation as continuation
from subnet import persistent_training_evidence as evidence
from ops.trainer_lifecycle import authorized_lr_evidence as helper


class CPU16Evidence(unittest.TestCase):
    def setUp(self):
        self.case = continuation.Continuation()
        self.addCleanup(self.case.doCleanups)
        self.case.setUp()

    def job(self, version=6):
        case = self.case
        declaration = dict(case.declaration,
            version=f'unaudited-training-execution-amendment-v{version}-' +
                    ('optimizer-cpu16' if version == 6 else 'objective-horizon'),
            method='fp32-task-gradient-effective-lr-objective-horizon-v1')
        declaration['training_objective'] = {
            'version': 'sequence-mean-preference-positive-nll-v1', 'positive_nll_weight': 1.0}
        declaration['training_horizon'] = dict(
            version='retained-adam-unit-positive-nll-16-v1', first_optimizer_step=1,
            updates=16, initial_checkpoint=case.f.cp['id'],
            initial_parent_descriptor_sha256=case.declaration['parent_descriptor_sha256'],
            coefficient_during=1.0, coefficient_after=0.0)
        if version == 6:
            declaration['optimizer_threading'] = dict(
                version='optimizer-only-cpu-threads-v1', steady_intraop=2,
                optimizer_intraop=16, interop=2)
        return dict(case.job, unaudited_training_execution=case.f.sign(declaration))

    def test_both_versions_recognize_rate_from_real_adam_continuation(self):
        for version in (5, 6):
            with self.subTest(version=version):
                # This unit tests the LR interpreter. The continuation fixture
                # has no unit-NLL forward telemetry, so it must not be passed
                # off as a full objective-report qualification.
                result = helper.expected_hyperparameters(
                    self.case.report, self.job(version), self.case.manifest, self.case.f.authority)
                self.assertEqual(result['lr'], 5e-7)

    def test_v6_does_not_fall_back_to_the_historical_default_rate(self):
        report = copy.deepcopy(self.case.report)
        report['training']['updates'][0]['hyperparameters']['lr'] = 1e-5
        with self.assertRaisesRegex(ValueError, 'attribution/counter/objective'):
            evidence.validate_updates(report, self.job(), self.case.manifest)

    def test_v6_keeps_original_counter_and_parent_guards(self):
        report = copy.deepcopy(self.case.report)
        report['training']['updates'][0]['global_optimizer_step'] = 3
        with self.assertRaisesRegex(ValueError, 'attribution/counter/objective'):
            evidence.validate_updates(report, self.job(), self.case.manifest)
        job = self.job()
        declaration = copy.deepcopy(job['unaudited_training_execution']['payload'])
        declaration['parent_descriptor_sha256'] = 'e' * 64
        job['unaudited_training_execution'] = self.case.f.sign(declaration)
        with self.assertRaisesRegex(ValueError, 'original ROOT job scope'):
            evidence.validate_updates(self.case.report, job, self.case.manifest)

    def test_v6_cannot_relabel_an_unsigned_declaration(self):
        job = self.job()
        job['unaudited_training_execution'].pop('signature')
        with self.assertRaises(Exception):
            evidence.validate_updates(self.case.report, job, self.case.manifest)

    def test_unknown_versions_are_not_interpreted_as_authorized_lr(self):
        self.assertNotIn('unaudited-training-execution-amendment-v7-unknown', helper.VERSIONS)


if __name__ == '__main__':
    unittest.main()
