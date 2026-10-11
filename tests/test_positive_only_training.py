"""CPU scientific controls; no production source dispatch or GPU qualification."""
import copy
import math
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from subnet import positive_only_training as sft
from subnet.fp32_gradient_accumulation import FP32GradientAccumulator
from subnet.persistent_cpu_adamw import HYPERPARAMETERS, parameter_inventory, sha

EOS = 2
SEED = 'a'*64
CHECKPOINT = 'b'*64
OBJECTIVE = dict(version=sft.OBJECTIVE_VERSION, completion_policy=sft.COMPLETION_POLICY,
                 task_weight_rule=sft.TASK_WEIGHT_RULE)


def pairs_for_tasks(count=2, quota=4):
    rows = []
    for task in range(count):
        definition = dict(env_id='math', spec={})
        for attempt in range(quota):
            def rollout(positive):
                return dict(env_id='math', index=task, task_hash=f'{task+1:064x}',
                    classification='positive' if positive else 'negative', done=True,
                    turns=[dict(prompt=[1, task+3],
                                output=[attempt+8, EOS] if positive else [29, attempt+8, EOS])])
            rows.append((definition, rollout(True), rollout(False)))
    return rows


class TinyModel(torch.nn.Module):
    def __init__(self, position_dependent=False):
        super().__init__()
        self.logit_weights = torch.nn.Parameter(
            torch.arange(256, dtype=torch.float32).reshape(8, 32)/100
            if position_dependent else torch.zeros(32))
        self.config = SimpleNamespace(attention_dropout=0)
        self.calls = []; self.checkpointing = False

    def forward(self, input_ids, *, use_cache):
        assert use_cache is False
        self.calls.append(input_ids.detach().tolist()[0])
        assert 29 not in self.calls[-1], 'negative response reached model'
        count = input_ids.shape[1]
        logits = (self.logit_weights[:count] if self.logit_weights.ndim == 2
                  else self.logit_weights.expand(count, -1))
        return SimpleNamespace(logits=logits.unsqueeze(0))

    def gradient_checkpointing_enable(self, **kwargs):
        self.checkpointing = True

    def gradient_checkpointing_disable(self):
        self.checkpointing = False

    def save_pretrained(self, destination, **kwargs):
        (Path(destination)/'model.safetensors').write_bytes(b'CPU lifecycle test only')


class TinyRuntime:
    def __init__(self):
        self.model = TinyModel()
        self.tokenizer = SimpleNamespace(eos_token_id=EOS,
            save_pretrained=lambda d: (Path(d)/'tokenizer.json').write_text('{}'))
        self.configured = []

    def configure(self, spec, harness):
        self.configured.append((spec, harness))


class FakePersistentOptimizer:
    """Lifecycle probe, not a replacement qualification for real Adam numerics."""
    instances = []

    def __init__(self, named_parameters, input_checkpoint, **kwargs):
        self.parameters = list(named_parameters)
        self.kwargs = kwargs; self.input_checkpoint = input_checkpoint
        self.global_step = 11
        self.hyperparameters = dict(HYPERPARAMETERS, lr=1e-6)
        self.instances.append(self)

    def zero_grad(self):
        for _, parameter in self.parameters:
            parameter.grad = None

    def step(self, *, gradients):
        assert set(gradients) == {n for n, _ in self.parameters}
        precision = []
        with torch.no_grad():
            for name, parameter in self.parameters:
                assert parameter.grad is None
                assert gradients[name].dtype == torch.float32
                before = parameter.detach().clone()
                parameter.sub_(self.hyperparameters['lr']*gradients[name])
                delta = parameter-before
                changed = int((delta != 0).sum())
                precision.append(dict(name=name, elements=parameter.numel(),
                    master_changed_elements=changed, bf16_changed_elements=changed,
                    master_delta_l2=float(torch.linalg.vector_norm(delta)),
                    master_delta_max_abs=float(delta.abs().max())))
        self.global_step += 1
        # Fake ABI fixture only; physical BF16/CUDA qualification is separate.
        return dict(optimizer_step=self.global_step, parameters=precision,
            master_changed_elements=sum(r['master_changed_elements'] for r in precision),
            bf16_changed_elements=sum(r['bf16_changed_elements'] for r in precision),
            optimizer_state_dtype='torch.float32', master_dtype='torch.float32',
            inference_dtype='torch.bfloat16', effective_hyperparameters=self.hyperparameters.copy(),
            learning_rate_authorization_sha256=sha(self.kwargs['learning_rate_authorization']))


class PositiveSelectionTests(unittest.TestCase):
    def test_complete_quota_preserved_before_sft_filter(self):
        pairs = pairs_for_tasks()
        pairs[0][1]['turns'][0]['output'][-1] = 20
        original = copy.deepcopy(pairs)
        positives, tasks, groups, selection = sft.select_positive_tasks(pairs, 2, SEED, EOS, 4)
        self.assertEqual(pairs, original)
        self.assertEqual(len(positives), 7)
        self.assertEqual(sorted(len(t['positive_indices']) for t in tasks), [3, 4])
        self.assertEqual(sorted(i for g in groups for i in g), [0, 1])
        self.assertEqual(selection['input_pair_count'], 8)
        self.assertEqual(selection['excluded'][0]['source_pair_index'], 0)
        self.assertEqual(selection['excluded'][0]['turn_reasons'],
                         [dict(turn_index=0, reason='missing-terminal-eos')])
        self.assertEqual(selection['generated_tokens'], 14)
        with self.assertRaisesRegex(ValueError, 'quota'):
            sft.select_positive_tasks(pairs[:-1], 1, SEED, EOS, 4)

    def test_zero_positive_task_is_recorded_and_not_repeated(self):
        pairs = pairs_for_tasks()
        for row in pairs[:4]:
            row[1]['turns'][0]['output'][-1] = 20
        _, tasks, groups, selection = sft.select_positive_tasks(pairs, 1, SEED, EOS, 4)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(groups, [[0]])
        self.assertEqual(selection['excluded_tasks'][0]['index'], 0)
        with self.assertRaisesRegex(ValueError, 'minibatches'):
            sft.select_positive_tasks(pairs, 2, SEED, EOS, 4)
        for row in pairs[4:]:
            row[1]['turns'][0]['output'][-1] = 20
        with self.assertRaisesRegex(ValueError, 'minibatches'):
            sft.select_positive_tasks(pairs, 1, SEED, EOS, 4)

    def test_exact_and_relabelled_positive_duplicates_refused(self):
        for exact in (True, False):
            pairs = pairs_for_tasks(1)
            if exact:
                pairs[1] = copy.deepcopy(pairs[0])
            else:
                pairs[1][1]['turns'] = copy.deepcopy(pairs[0][1]['turns'])
                pairs[1][1]['metadata'] = 'different label cannot add diversity'
            with self.subTest(exact=exact), self.assertRaisesRegex(ValueError, 'duplicate|shared rollout'):
                sft.select_positive_tasks(pairs, 1, SEED, EOS, 4)

    def test_interior_eos_and_nonterminal_turn_excluded(self):
        pairs = pairs_for_tasks(1)
        pairs[0][1]['turns'][0]['output'] = [EOS, 8, EOS]
        pairs[1][1]['turns'].append(dict(prompt=[1], output=[9]))
        _, _, _, selection = sft.select_positive_tasks(pairs, 1, SEED, EOS, 4)
        self.assertEqual(len(selection['excluded']), 2)
        self.assertEqual(selection['excluded'][0]['turn_reasons'][0]['reason'], 'eos-before-terminal')
        self.assertEqual(selection['excluded'][1]['turn_reasons'][0]['turn_index'], 1)

    def test_invalid_nonce_like_indices_and_eos_types_rejected(self):
        for value in (-1, True, 2.0, None):
            with self.subTest(eos=value), self.assertRaises(ValueError):
                sft.select_positive_tasks(pairs_for_tasks(1), 1, SEED, value, 4)
        for value in (-1, True):
            pairs = pairs_for_tasks(1)
            pairs[0][1]['index'] = value
            with self.subTest(index=value), self.assertRaises(ValueError):
                sft.select_positive_tasks(pairs, 1, SEED, EOS, 4)
        pairs = pairs_for_tasks(1); pairs[0][1]['turns'][0]['output'][0] = True
        with self.assertRaisesRegex(ValueError, 'integer'):
            sft.select_positive_tasks(pairs, 1, SEED, EOS, 4)

    def test_deterministic_once_through_uneven_partition(self):
        args = (pairs_for_tasks(5), 3, SEED, EOS, 4)
        first = sft.select_positive_tasks(*args)
        self.assertEqual(first, sft.select_positive_tasks(*args))
        self.assertEqual([len(g) for g in first[2]], [2, 2, 1])
        self.assertEqual([i for g in first[2] for i in g], list(range(5)))
        with self.assertRaises(ValueError):
            sft.select_positive_tasks(*args[:-1], None)


class ObjectiveMathTests(unittest.TestCase):
    def test_prompt_mask_next_token_alignment_and_eos_in_loss(self):
        model = TinyModel(position_dependent=True)
        rollout = dict(turns=[dict(prompt=[1, 3, 4], output=[8, EOS])])
        value = sft.sequence_mean_logprob(torch, model, rollout, device='cpu')
        expected = (torch.log_softmax(model.logit_weights[2], -1)[8] +
                    torch.log_softmax(model.logit_weights[3], -1)[EOS])/2
        torch.testing.assert_close(value, expected)
        (-value).backward()
        nonzero_rows = (model.logit_weights.grad.abs().sum(-1) > 0).nonzero().flatten().tolist()
        self.assertEqual(nonzero_rows, [2, 3])
        self.assertLess(float(model.logit_weights.grad[3, EOS]), 0)

    def test_multiturn_total_generated_token_mean(self):
        model = TinyModel(position_dependent=True)
        rollout = dict(turns=[dict(prompt=[1], output=[8, EOS]),
                             dict(prompt=[1, 3], output=[9, 10, EOS])])
        actual = sft.sequence_mean_logprob(torch, model, rollout, device='cpu')
        lp = torch.log_softmax(model.logit_weights, -1)
        expected = (lp[0, 8]+lp[1, EOS]+lp[1, 9]+lp[2, 10]+lp[3, EOS])/5
        torch.testing.assert_close(actual, expected)

    def test_three_and_four_responses_have_equal_task_gradient_weight(self):
        p = torch.nn.Parameter(torch.tensor(1.))
        factors = [1., 2., 3., 8., 9., 10., 11.]
        tasks = [dict(task_sha256='a', positive_indices=[0, 1, 2]),
                 dict(task_sha256='b', positive_indices=[3, 4, 5, 6])]
        acc = FP32GradientAccumulator([('p', p)])
        rows = sft.accumulate_positive_tasks(torch, lambda i: -p*factors[i],
            tasks, [0, 1], after_backward=acc.capture)
        self.assertEqual(acc.microsteps, 7)
        self.assertIsNone(p.grad)
        self.assertAlmostEqual(float(acc.buffers['p']), (2.+9.5)/2, places=6)
        self.assertAlmostEqual(sum(r['weight'] for r in rows[:3]), .5)
        self.assertAlmostEqual(sum(r['weight'] for r in rows[3:]), .5)
        components = [dict(positive_index=i, positive_mean_logprob=-v, positive_nll=v)
                      for i, v in enumerate(factors)]
        self.assertAlmostEqual(sft.weighted_positive_summary(components, tasks)['positive_nll'], 5.75)

    def test_invalid_group_indices_and_nonfinite_values_refused(self):
        tasks = [dict(task_sha256='a', positive_indices=[0])]
        for indices in ([True], [-1], [1], [0, 0], []):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                sft.accumulate_positive_tasks(torch, lambda i: torch.tensor(-1., requires_grad=True),
                    tasks, indices, after_backward=lambda: None)
        for value in (torch.tensor(float('nan')), torch.tensor([-1.])):
            with self.assertRaises(ValueError):
                sft.capture_positive_components(torch, lambda i: value, 1)

    def test_kernel_partition_rejects_duplicate_and_boolean_rows(self):
        for indices in ([0, 0], [True], [-1], []):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                sft.fixed_task_minibatches([dict(task_sha256='a', positive_indices=indices)], 1)
        with self.assertRaisesRegex(ValueError, 'only once'):
            sft.fixed_task_minibatches([dict(task_sha256='a', positive_indices=[0]),
                                       dict(task_sha256='b', positive_indices=[0])], 1)


class EpochLifecycleTests(unittest.TestCase):
    def setUp(self):
        FakePersistentOptimizer.instances.clear()

    def test_full_fake_runtime_never_scores_negative_preserves_state_and_lr(self):
        runtime = TinyRuntime(); pairs = pairs_for_tasks(2)
        pairs[0][1]['turns'][0]['output'][-1] = 20
        restored = object(); authorization = dict(test='authorized effective LR')
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(sft, '_admit_model_profile',
                side_effect=lambda torch, model: list(model.parameters())))
            stack.enter_context(patch.object(sft, 'admit_capacity', return_value=dict(admitted=True)))
            stack.enter_context(patch.object(sft, 'PersistentCPUAdamW', FakePersistentOptimizer))
            stack.enter_context(patch('subnet.protocol.harness_for', return_value={}))
            for name in ('synchronize', 'reset_peak_memory_stats', 'max_memory_allocated',
                         'max_memory_reserved', 'empty_cache'):
                stack.enter_context(patch.object(torch.cuda, name, return_value=0))
            path, optimizer, diagnostics = sft.train_epoch(runtime, pairs, directory,
                input_checkpoint=CHECKPOINT, epoch='prospective-cpu-test', seed=SEED, steps=2,
                restored_state=restored, resource_admission=dict(admitted=True),
                required_pairs_per_task=4, learning_rate_authorization=authorization,
                learning_rate_authority='cpu-test-authority', job_id='cpu-test', objective=OBJECTIVE)
            self.assertTrue(path.is_dir())
            self.assertIs(optimizer.kwargs['restored'], restored)
            self.assertIsNone(optimizer.kwargs['approved_genesis'])
            self.assertEqual(optimizer.kwargs['learning_rate_authorization'], authorization)
            self.assertEqual(diagnostics['global_optimizer_step_before'], 11)
            self.assertEqual(diagnostics['global_optimizer_step_after'], 13)
            self.assertEqual(diagnostics['effective_hyperparameters']['lr'], 1e-6)
            self.assertEqual(diagnostics['positive_count'], 7)
            self.assertEqual(len(runtime.model.calls), 7*3)  # before, backward, after
            self.assertTrue(all(29 not in tokens and 20 not in tokens for tokens in runtime.model.calls))
            self.assertEqual(sum(r['gradient_positives'] for r in diagnostics['updates']), 7)
            self.assertEqual(sorted(r['positive_index'] for u in diagnostics['updates']
                                    for r in u['positives']), list(range(7)))
            self.assertTrue(all(u['gradient_accumulation_dtype'] == 'torch.float32'
                                for u in diagnostics['updates']))
            self.assertTrue(diagnostics['positive_nll_components']['no_negative_forwards'])
            self.assertNotIn('training_pair_margin_before', diagnostics)
            self.assertNotIn('gradient_pairs', diagnostics['updates'][0])
            self.assertFalse(diagnostics['complete'])
            self.assertFalse(diagnostics['heldout_gain_claimed'])
            self.assertFalse(runtime.model.checkpointing)
            from subnet.positive_only_training_evidence import validate_diagnostics
            from subnet.persistent_training_worker import report_updates
            updates, summary = report_updates(diagnostics,
                dict(steps=2, training_policy=sft.POLICY),
                dict(epoch='prospective-cpu-test', checkpoint=dict(id=CHECKPOINT)))
            expectations = dict(objective=OBJECTIVE, input_checkpoint=CHECKPOINT,
                epoch='prospective-cpu-test', steps=2, seed=SEED, global_step_before=11,
                parameters=parameter_inventory(runtime.model.named_parameters())[1],
                eos_token_id=EOS, required_pairs_per_task=4,
                effective_hyperparameters=dict(HYPERPARAMETERS, lr=1e-6),
                learning_rate_authorization_sha256=sha(authorization))
            checked = validate_diagnostics(summary, updates, pairs, **expectations)
            self.assertEqual(checked['distinct_positives'], 7)
            self.assertFalse(checked['historical_gpu_execution_proven'])
            for mutation in ('weight', 'pair-field', 'eos', 'wrong-lr'):
                changed_summary = copy.deepcopy(summary); changed_updates = copy.deepcopy(updates)
                if mutation == 'weight':
                    changed_updates[0]['positives'][0]['gradient_weight'] *= 2
                elif mutation == 'pair-field':
                    changed_updates[0]['gradient_pairs'] = 4
                elif mutation == 'eos':
                    changed_summary['eos_token_id'] = 20
                else:
                    changed_updates[0]['hyperparameters']['lr'] = 1e-5
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    validate_diagnostics(changed_summary, changed_updates, pairs, **expectations)

    def test_no_eligible_positives_fails_before_optimizer_or_model_work(self):
        runtime = TinyRuntime(); pairs = pairs_for_tasks(1)
        for row in pairs:
            row[1]['turns'][0]['output'][-1] = 20
        with patch.object(sft, 'PersistentCPUAdamW') as constructor, tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                sft.train_epoch(runtime, pairs, directory, input_checkpoint=CHECKPOINT,
                    epoch='test', seed=SEED, steps=1, resource_admission={},
                    required_pairs_per_task=4, objective=OBJECTIVE)
            constructor.assert_not_called()
            self.assertEqual(runtime.model.calls, [])

    def test_actual_production_profile_refuses_cpu(self):
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            sft._admit_model_profile(torch, TinyModel())


if __name__ == '__main__':
    unittest.main()
