"""Prospective objective dispatch; no service or model execution."""
import copy
import sys
import types
import unittest
from unittest.mock import patch

from subnet.positive_only_objective import VERSION, configuration
from subnet.training_policy import OBJECTIVE_VERSION, objective_config, objective_at_step
from subnet.persistent_training_worker import objective_trainer, validate_objective_runtime
from subnet.unaudited_training_execution import configured_objective, OBJECTIVE_VERSIONS


class PositiveOnlyObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.objective = dict(version=VERSION, completion_policy='terminal-eos-v1',
            task_weight_rule='mean-positive-within-task-then-mean-task-within-group')

    def test_configuration_is_explicit_and_copied(self):
        self.assertEqual(objective_config(self.objective), self.objective)
        result = configuration(self.objective)
        result['completion_policy'] = 'all-answers'
        self.assertEqual(self.objective['completion_policy'], 'terminal-eos-v1')

    def test_implicit_or_changed_semantics_rejected(self):
        alternatives = [None, {}, dict(self.objective, positive_nll_weight=1),
            dict(self.objective, completion_policy='boxed-at-cap'),
            dict(self.objective, task_weight_rule='mean-all-tokens'),
            dict(self.objective, negative_weight=0)]
        for value in alternatives:
            with self.subTest(value=value), self.assertRaises(ValueError):
                configuration(value)

    def test_zero_and_unit_remain_preference_not_sft(self):
        from subnet.task_normalized_training import train_epoch
        self.assertEqual(objective_trainer(None), (train_epoch, {'positive_nll_weight': 0.0}))
        for coefficient in (0, 1):
            value = dict(version=OBJECTIVE_VERSION, positive_nll_weight=coefficient)
            self.assertEqual(objective_trainer(value),
                (train_epoch, {'positive_nll_weight': coefficient}))

    def test_sft_dispatch_never_invokes_preference_kernel(self):
        stub = types.ModuleType('subnet.positive_only_training')
        stub.train_epoch = object()
        with patch.dict(sys.modules, {'subnet.positive_only_training': stub}):
            fn, arguments = objective_trainer(self.objective)
        self.assertIs(fn, stub.train_epoch)
        self.assertEqual(arguments, {'objective': self.objective})

    def test_old_execution_release_cannot_authorize_new_sft(self):
        # Every existing execution kind still refuses this configuration. A new
        # signed qualification/release is necessary before production selection.
        for version in [*OBJECTIVE_VERSIONS, 'historical-other-execution']:
            declaration = dict(version=version, training_objective=self.objective)
            with self.subTest(version=version), self.assertRaises(ValueError):
                configured_objective(declaration)

    def test_complete_old_horizon_still_cannot_be_reinterpreted_as_sft(self):
        from subnet.training_policy import HORIZON_VERSION
        from subnet.unaudited_training_execution import OBJECTIVE_VERSION as EXECUTION, OBJECTIVE_RELEASE
        horizon = dict(version=HORIZON_VERSION, first_optimizer_step=26, updates=16,
            initial_checkpoint='a'*64, initial_parent_descriptor_sha256='b'*64,
            coefficient_during=1.0, coefficient_after=0.0)
        for version in (EXECUTION, OBJECTIVE_RELEASE):
            declaration = dict(version=version, training_horizon=horizon, steps=1,
                effective_learning_rate=5e-7, optimizer_step_before=26,
                minimum_optimizer_step=26, parent_descriptor_sha256='b'*64,
                training_objective=dict(version=OBJECTIVE_VERSION, positive_nll_weight=1.0))
            self.assertEqual(configured_objective(declaration), declaration['training_objective'])
            declaration['training_objective'] = self.objective
            with self.subTest(version=version), self.assertRaises(ValueError):
                configured_objective(declaration)

    def test_unknown_objective_never_defaults_to_legacy(self):
        with self.assertRaises(ValueError):
            objective_trainer(dict(version='unknown'))

    def test_sft_tokenizer_must_match_independently_signed_eos(self):
        runtime=types.SimpleNamespace(tokenizer=types.SimpleNamespace(eos_token_id=151645))
        validate_objective_runtime(self.objective, {'tokenizer_eos_token_id':151645}, runtime)
        for pinned in (None, True, -1, 151643):
            with self.subTest(pinned=pinned), self.assertRaises(ValueError):
                validate_objective_runtime(self.objective, {'tokenizer_eos_token_id':pinned}, runtime)
        runtime.tokenizer.eos_token_id=True
        with self.assertRaises(ValueError):
            validate_objective_runtime(self.objective, {'tokenizer_eos_token_id':151645}, runtime)

    def test_legacy_runtime_does_not_acquire_new_eos_requirement(self):
        runtime=object()
        validate_objective_runtime(None,{},runtime)
        validate_objective_runtime(dict(version=OBJECTIVE_VERSION, positive_nll_weight=0),{},runtime)


if __name__ == '__main__':
    unittest.main()
