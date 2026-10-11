"""Actual CPU SFT diagnostics through the common signed-report dispatcher.

The tiny runtime executes real autograd and FP32 gradient accumulation. Its
optimizer/precision receipt is an explicit fake ABI fixture, not GPU or release
qualification. Only throwaway signing keys are used; nothing is deployed.
"""
import base64
import copy
import json
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'tests')]

import torch
from nacl.signing import SigningKey
from subnet import positive_only_training as sft
from subnet.learning_rate_transition import VERSION as LR_VERSION
from subnet.persistent_cpu_adamw import HYPERPARAMETERS, parameter_inventory, sha
from subnet.persistent_training_evidence import validate_updates
from subnet.persistent_training_worker import report_updates
from subnet.storage import canonical
from subnet.unaudited_training_execution import SFT_METHOD, SFT_VERSION
from test_positive_only_training import (
    CHECKPOINT, EOS, OBJECTIVE, SEED, FakePersistentOptimizer, TinyRuntime,
    pairs_for_tasks,
)


class ActualCPUPositiveEvidenceDispatchTests(unittest.TestCase):
    @classmethod
    def sign(cls, value):
        return dict(payload=copy.deepcopy(value), signer=cls.authority,
            signature=base64.b64encode(cls.key.sign(canonical(value)).signature).decode())

    @classmethod
    def setUpClass(cls):
        cls.key = SigningKey.generate()
        cls.authority = cls.key.verify_key.encode().hex()
        runtime = TinyRuntime()
        pairs = pairs_for_tasks(2, quota=4)
        # One capped positive remains in the admitted K4/L4 input but is excluded
        # by the trainer. This makes EOS selection observable at the boundary.
        pairs[0][1]['turns'][0]['output'][-1] = 20
        inventory = parameter_inventory(runtime.model.named_parameters())[1]
        parameters_sha = sha(inventory)
        epoch, job_id = 'cpu-sft-dispatch', 'cpu-retained-adam-dispatch'
        binding = dict(input_checkpoint=CHECKPOINT, global_step_before=11,
            parameters=inventory, parameters_sha256=parameters_sha,
            parent={'descriptor_sha256': 'c'*64}, genesis_sha256='d'*64)
        manifest = dict(epoch=epoch, checkpoint={'id': CHECKPOINT},
            trainer_state_binding=binding, environments=[pairs[0][0]], K=4, L=4,
            training_coverage={'seed': SEED})
        grant = dict(version=LR_VERSION, epoch=epoch, job_id=job_id,
            input_checkpoint=CHECKPOINT, parent_descriptor_sha256='c'*64,
            genesis_sha256='d'*64, optimizer_step_before=11, steps=2,
            parameters_sha256=parameters_sha,
            base_hyperparameters_sha256=sha(HYPERPARAMETERS),
            effective_learning_rate=1e-6, created_at=1, expires_at=100,
            execution_release_sha256='e'*64)
        signed_grant = cls.sign(grant)
        declaration = dict(version=SFT_VERSION, method=SFT_METHOD, epoch=epoch,
            job_id=job_id, steps=2, effective_learning_rate=1e-6,
            training_objective=OBJECTIVE, tokenizer_eos_token_id=EOS,
            learning_rate_authorization=signed_grant)
        job = dict(job_id=job_id, manifest=cls.sign(manifest), steps=2,
            training_policy=sft.POLICY,
            training_input_policy='committed-unaudited-training-v1',
            unaudited_training_execution=cls.sign(declaration))
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(sft, '_admit_model_profile',
                side_effect=lambda torch, model: list(model.parameters())))
            stack.enter_context(patch.object(sft, 'admit_capacity',
                return_value=dict(admitted=True)))
            stack.enter_context(patch.object(sft, 'PersistentCPUAdamW', FakePersistentOptimizer))
            stack.enter_context(patch('subnet.protocol.harness_for', return_value={}))
            for name in ('synchronize', 'reset_peak_memory_stats',
                         'max_memory_allocated', 'max_memory_reserved', 'empty_cache'):
                stack.enter_context(patch.object(torch.cuda, name, return_value=0))
            _, optimizer, diagnostics = sft.train_epoch(runtime, pairs, directory,
                input_checkpoint=CHECKPOINT, epoch=epoch, seed=SEED, steps=2,
                restored_state=object(), resource_admission={'admitted': True},
                required_pairs_per_task=4, learning_rate_authorization=signed_grant,
                learning_rate_authority=cls.authority, job_id=job_id, objective=OBJECTIVE)
            updates, summary = report_updates(diagnostics, job, manifest)
        batches = []
        for task_index in range(2):
            selected = [row for row in pairs if row[1]['index'] == task_index]
            batches.append(dict(env_id='math', index=task_index,
                rollouts=[row[1] for row in selected]+[row[2] for row in selected]))
        cls.original_report = dict(training={'updates': updates,
            'persistent_diagnostics': summary},
            training_admissions=[{'claimed_batch': batch} for batch in batches])
        cls.original_job = job
        cls.original_manifest = manifest
        cls.original_declaration = declaration
        cls.actual_calls = copy.deepcopy(runtime.model.calls)
        cls.actual_step = optimizer.global_step

    def setUp(self):
        self.report = copy.deepcopy(self.original_report)
        self.job = copy.deepcopy(self.original_job)
        self.manifest = copy.deepcopy(self.original_manifest)
        self.declaration = copy.deepcopy(self.original_declaration)

    def test_actual_train_report_dispatches_with_retained_state_and_signed_rate(self):
        result = validate_updates(self.report, self.job, self.manifest)
        self.assertEqual(result['distinct_positives'], 7)
        self.assertEqual(result['excluded_positives'], 1)
        self.assertEqual(result['distinct_tasks'], 2)
        self.assertEqual((result['global_step_before'], result['global_step_after']), (11, 13))
        self.assertEqual(self.actual_step, 13)
        self.assertEqual(len(self.actual_calls), 21)
        self.assertTrue(all(29 not in tokens and 20 not in tokens for tokens in self.actual_calls))
        self.assertFalse(result['historical_gpu_execution_proven'])
        self.assertFalse(result['heldout_gain_claimed'])

    def test_report_cannot_replace_the_signed_eos(self):
        diagnostics = self.report['training']['persistent_diagnostics']
        diagnostics['eos_token_id'] = 20
        diagnostics['positive_nll_components']['eos_token_id'] = 20
        diagnostics['positive_selection']['eos_token_id'] = 20
        with self.assertRaisesRegex(ValueError, 'EOS|eos|selection'):
            validate_updates(self.report, self.job, self.manifest)

    def test_new_signed_eos_still_requires_matching_actual_selection(self):
        self.declaration['tokenizer_eos_token_id'] = 20
        self.job['unaudited_training_execution'] = self.sign(self.declaration)
        with self.assertRaises(ValueError):
            validate_updates(self.report, self.job, self.manifest)

    def test_consistently_changed_report_rate_cannot_override_grant(self):
        diagnostics = self.report['training']['persistent_diagnostics']
        diagnostics['effective_hyperparameters']['lr'] = 5e-7
        for update in self.report['training']['updates']:
            update['hyperparameters']['lr'] = 5e-7
            update['precision']['effective_hyperparameters']['lr'] = 5e-7
        with self.assertRaisesRegex(ValueError, 'hyperparameters|rate'):
            validate_updates(self.report, self.job, self.manifest)

    def test_signed_grant_rate_must_agree_with_execution_declaration(self):
        grant = copy.deepcopy(self.declaration['learning_rate_authorization']['payload'])
        grant['effective_learning_rate'] = 5e-7
        self.declaration['learning_rate_authorization'] = self.sign(grant)
        self.job['unaudited_training_execution'] = self.sign(self.declaration)
        with self.assertRaisesRegex(ValueError, 'signed rate disagree'):
            validate_updates(self.report, self.job, self.manifest)

    def test_signed_grant_cannot_be_transplanted_from_another_checkpoint(self):
        grant = copy.deepcopy(self.declaration['learning_rate_authorization']['payload'])
        grant['input_checkpoint'] = 'f'*64
        self.declaration['learning_rate_authorization'] = self.sign(grant)
        self.job['unaudited_training_execution'] = self.sign(self.declaration)
        with self.assertRaisesRegex(ValueError, 'parent/job/input/counter binding'):
            validate_updates(self.report, self.job, self.manifest)

    def test_manifest_argument_is_authenticated_not_chosen_by_report(self):
        self.manifest['training_coverage']['seed'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'authenticated report manifest'):
            validate_updates(self.report, self.job, self.manifest)

    def test_resigned_coverage_seed_cannot_reuse_original_task_schedule(self):
        self.manifest['training_coverage']['seed'] = 'f'*64
        self.job['manifest'] = self.sign(self.manifest)
        with self.assertRaisesRegex(ValueError, 'coverage|selection|attribution|tasks'):
            validate_updates(self.report, self.job, self.manifest)

    def test_changed_admitted_trajectory_cannot_reuse_original_diagnostics(self):
        batch = self.report['training_admissions'][0]['claimed_batch']
        batch['rollouts'][1]['turns'][0]['output'][0] = 21
        with self.assertRaisesRegex(ValueError, 'selection|attribution'):
            validate_updates(self.report, self.job, self.manifest)

    def test_outer_saved_dispatch_refuses_resigned_changed_environment_spec(self):
        # Exercise the real role-independent RemoteJobs guard with the existing
        # metadata fixture. This is not a full SFT release/admission fixture.
        from test_remote_backend import RemoteReportBinding
        fixture = RemoteReportBinding('test_approved_report_binding')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.manifest['environments'] = copy.deepcopy(self.manifest['environments'])
        fixture.prior['manifest_sha256'] = sha(fixture.manifest)

        def fixture_sign(value):
            return dict(payload=copy.deepcopy(value), signer=fixture.operator,
                signature=base64.b64encode(
                    fixture.key.sign(canonical(value)).signature).decode())

        original_job = dict(fixture.envelope['payload'],
                            manifest=fixture_sign(fixture.manifest))
        fixture.envelope = fixture_sign(original_job)
        fixture.prior['job_sha256'] = fixture.report['job_sha256'] = sha(original_job)
        job_path = fixture.jobs.state/'same-job-job.json'
        job_path.write_text(json.dumps(fixture.envelope))
        self.assertIs(fixture.jobs.checked(fixture.report, fixture.prior, fixture.manifest),
                      fixture.report)

        changed_manifest = copy.deepcopy(fixture.manifest)
        changed_manifest['environments'][0]['spec']['revision'] = 'another-task-source'
        # Even another valid authority signature cannot change what the already
        # dispatched original job/manifest was. The stored prior stays fixed.
        changed_job = dict(original_job, manifest=fixture_sign(changed_manifest))
        job_path.write_text(json.dumps(fixture_sign(changed_job)))
        with patch('subnet.unaudited_training_execution.required_report') as downstream:
            with self.assertRaisesRegex(ValueError, 'remote role report binding'):
                fixture.jobs.checked(fixture.report, fixture.prior, changed_manifest)
            downstream.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
