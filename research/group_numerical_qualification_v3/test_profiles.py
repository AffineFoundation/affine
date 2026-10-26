"""Tiny CPU causal-model controls; no external model, GPU or network calls."""
from dataclasses import replace
from types import SimpleNamespace
import math
import unittest
from unittest.mock import patch

import torch

from research.fixed_group_grpo.core import Group, Policy, Rollout, Stamp, policy_probabilities
from .profiles import (buffer_inventory, cached_replay, compare_rows, drift, prefill,
                       prepare_dtype, preserve_original_rope)


class Cache:
    def __init__(self, length, sums): self.length, self.sums = length, sums
    def get_seq_length(self): return self.length


class CausalToy(torch.nn.Module):
    """All past unmasked tokens influence the next-token distribution."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(64, dtype=torch.float32).reshape(8, 8)/37)
        self.position_weight = torch.nn.Parameter(torch.linspace(-.09, .12, 8))
        self.config = SimpleNamespace(model_type='qwen2', vocab_size=8,
                                      max_position_embeddings=16, attention_dropout=0.)
        self.checkpointing = False
        self.calls = []
    def gradient_checkpointing_disable(self): self.checkpointing = False
    def gradient_checkpointing_enable(self, **kwargs):
        assert kwargs == {'gradient_checkpointing_kwargs': {'use_reentrant': False}}
        self.checkpointing = True
    def forward(self, input_ids, attention_mask, use_cache, position_ids=None,
                past_key_values=None, logits_to_keep=None):
        width = input_ids.shape[1]
        visible = attention_mask[:, -width:]
        old = (past_key_values.sums if past_key_values is not None else
               torch.zeros(input_ids.shape[0], dtype=torch.long))
        sums = (input_ids*visible).cumsum(-1)+old[:, None]
        positions = ((attention_mask.cumsum(-1)-1).clamp_min(0)[:, -width:]
                     if position_ids is None else position_ids)
        logits = self.weight[sums % 8] + positions[:, :, None]*self.position_weight
        self.calls.append(dict(ids=input_ids.detach().clone(), mask=attention_mask.detach().clone(),
                               positions=positions.detach().clone(), cache=use_cache,
                               grad=torch.is_grad_enabled(), train=self.training,
                               checkpointing=self.checkpointing))
        length = (past_key_values.length if past_key_values is not None else 0)+width
        return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits,
                               past_key_values=Cache(length, sums[:, -1].detach()) if use_cache else None)


def fixture():
    policy = Policy(.7, 8, 7, 6, 16)
    stamp = Stamp('fixed', 'a'*64, 0, 'b'*64, policy)
    outputs = [(7,), (1, 7), (2, 3, 7), (1, 2, 3, 7), (6, 5, 4, 3, 7),
               (4, 4, 4, 4, 4, 4), (1, 7), (3, 4, 5, 6, 7)]
    rows = [Rollout(i, output, tuple(-1. for _ in output),
                    'budget' if i == 5 else 'eos', 'unresolved' if i == 5 else 'wrong')
            for i, output in enumerate(outputs)]
    return Group(stamp, 'fixed-task', (1, 2, 3), tuple(rows))


class Profiles(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model, self.group = CausalToy(), fixture()

    def test_next_token_alignment_matches_manual_prefix(self):
        values, timing = cached_replay(self.model, self.group)
        self.assertEqual(timing['forward_calls'], 6)
        for row, actual in zip(self.group.rollouts, values):
            expected = []
            for position, token in enumerate(row.output):
                prefix = self.group.prompt+row.output[:position]
                logits = self.model.weight[sum(prefix) % 8] + (len(prefix)-1)*self.model.position_weight
                expected.append(float(policy_probabilities(logits, self.group.stamp.policy)[token].log().detach()))
            torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), atol=0., rtol=0.)

    def test_ended_rows_are_masked_and_eos_is_scored_once(self):
        values, _ = cached_replay(self.model, self.group)
        self.assertEqual(list(map(len, values)), [1, 2, 3, 4, 5, 6, 2, 5])
        calls = self.model.calls
        self.assertEqual(tuple(calls[0]['ids'].shape), (8, 3))
        self.assertTrue(all(tuple(c['ids'].shape) == (8, 1) for c in calls[1:]))
        self.assertEqual(calls[1]['ids'][0, 0].item(), 7)
        self.assertEqual(calls[1]['mask'][0].tolist(), [1, 1, 1, 0])
        self.assertEqual(calls[-1]['mask'][0].tolist(), [1, 1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(calls[-1]['positions'][5].tolist(), [7])
        self.assertTrue(all(not c['grad'] and not c['train'] and not c['checkpointing'] for c in calls))

    def test_prefill_modes_match_causal_replay_without_parameter_updates(self):
        before = [p.detach().clone() for p in self.model.parameters()]
        cached, _ = cached_replay(self.model, self.group)
        for gradients in (False, True):
            self.model.calls.clear()
            values, timing = prefill(self.model, self.group, gradients=gradients)
            self.assertEqual(timing['forward_calls'], 8)
            for expected, actual in zip(cached, values):
                torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=0., atol=0.)
            self.assertTrue(all(c['grad'] == gradients and c['train'] == gradients
                                and c['checkpointing'] == gradients and not c['cache']
                                for c in self.model.calls))
        for old, now in zip(before, self.model.parameters()):
            self.assertTrue(torch.equal(old, now.detach())); self.assertIsNone(now.grad)

    def test_no_random_sampling_or_outcome_filtering(self):
        # All-failure, unresolved and identical-text attempts must all survive.
        with patch('research.fixed_group_grpo.core.draw_and_record', side_effect=AssertionError('no draws')):
            values, _ = cached_replay(self.model, self.group)
        self.assertEqual(len(values), 8)
        self.assertEqual(values[1], values[6])
        report = compare_rows(self.group, values, values)
        self.assertEqual(report['all_tokens']['tokens'], sum(len(r.output) for r in self.group.rollouts))
        self.assertEqual(report['rows'][5]['original_stop'], 'budget')
        self.assertEqual(report['rows'][5]['original_tokens'], 6)

    def test_rejects_post_eos_and_missing_attempt_before_forward(self):
        row = replace(self.group.rollouts[0], output=(7, 1), behavior_logprobs=(-1., -1.))
        for group in (replace(self.group, rollouts=(row,)+self.group.rollouts[1:]),
                      replace(self.group, rollouts=self.group.rollouts[:-1])):
            with self.assertRaises(ValueError): cached_replay(self.model, group)
        self.assertEqual(self.model.calls, [])

    def test_guard_checks_every_cached_position(self):
        calls = []
        def guard():
            calls.append(1)
            if len(calls) == 3: raise RuntimeError('diagnostic deadline')
        with self.assertRaisesRegex(RuntimeError, 'deadline'):
            cached_replay(self.model, self.group, guard=guard)
        self.assertEqual(len(self.model.calls), 2)

    def test_dropout_rejected_for_both_profiles(self):
        self.model.config.attention_dropout = .1
        with self.assertRaisesRegex(ValueError, 'dropout'):
            cached_replay(self.model, self.group)
        with self.assertRaisesRegex(ValueError, 'dropout'):
            prefill(self.model, self.group, gradients=False)


class Statistics(unittest.TestCase):
    def test_signed_asymmetric_ppo_clip_and_no_admission_threshold(self):
        changes = [math.log(.79), math.log(.82), 0., math.log(1.19), math.log(1.21)]
        result = drift([[-10.]*5], [[-10.+v for v in changes]])
        self.assertEqual(result['clipped_lower_tokens'], 1)
        self.assertEqual(result['clipped_upper_tokens'], 1)
        self.assertEqual(result['clipped_fraction'], .4)
        self.assertAlmostEqual(result['max_abs_logratio'], abs(math.log(.79)))
        self.assertFalse(result['qualification_pass_claimed'])
        self.assertTrue(result['diagnostic_only'])

    def test_shape_and_nonfinite_rejected(self):
        for a, b in [([[-1.]], [[-1., -2.]]), ([[-1.]], [[float('nan')]]), ([], [])]:
            with self.assertRaises(ValueError): drift(a, b)


class Dtypes(unittest.TestCase):
    def model(self):
        model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        model.register_buffer('inv_freq', torch.tensor([.1234567, .7654321], dtype=torch.float32))
        return model

    def test_bf16_preserves_fp32_rope_buffers(self):
        model = self.model(); before = buffer_inventory(model)
        prepare_dtype(model, torch.bfloat16, device='cpu')
        self.assertEqual(model.weight.dtype, torch.bfloat16)
        self.assertEqual(model.inv_freq.dtype, torch.float32)
        self.assertEqual(buffer_inventory(model), before)

    def test_fp32_widens_parameters_without_rounding_original_buffers(self):
        model = self.model(); before = buffer_inventory(model); expected = model.weight.detach().float().clone()
        prepare_dtype(model, torch.float32, device='cpu')
        self.assertEqual(model.weight.dtype, torch.float32)
        self.assertTrue(torch.equal(expected, model.weight.detach()))
        self.assertEqual(buffer_inventory(model), before)


class RopeConfiguration(unittest.TestCase):
    def build(self):
        class Config:
            def __init__(self):
                self.rope_parameters = {'rope_type': 'default', 'rope_theta': 10000.}
                self.max_position_embeddings = 4096
            @property
            def rope_scaling(self): return self.rope_parameters
        raw = dict(model_type='qwen2', rope_theta=10000., max_position_embeddings=4096)
        return Config(), raw

    def test_normalized_default_alias_is_preserved(self):
        config, raw = self.build(); identity = config.rope_parameters
        report = preserve_original_rope(config, raw, declared_context=8192)
        self.assertIs(config.rope_parameters, identity)
        self.assertEqual(config.max_position_embeddings, 8192)
        self.assertFalse(report['new_scaling_applied'])

    def test_installed_qwen2_config_normalization_without_model_construction(self):
        import importlib.metadata
        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
        self.assertEqual(importlib.metadata.version('transformers'), '5.14.1')
        _, raw = self.build()
        config = Qwen2Config(**raw)
        self.assertEqual(config.rope_scaling, {'rope_type': 'default', 'rope_theta': 10000.})
        preserve_original_rope(config, raw, declared_context=8192)
        self.assertEqual(config.rope_parameters, {'rope_type': 'default', 'rope_theta': 10000.})

    def test_dynamic_theta_factor_or_context_change_is_rejected(self):
        for field, value in [('rope_type', 'dynamic'), ('rope_theta', 1000000.), ('factor', 2.)]:
            config, raw = self.build(); config.rope_parameters[field] = value
            with self.assertRaises(ValueError): preserve_original_rope(config, raw, declared_context=8192)
        config, raw = self.build(); config.max_position_embeddings = 16384
        with self.assertRaises(ValueError): preserve_original_rope(config, raw, declared_context=8192)

    def test_unexpected_raw_scaling_is_rejected(self):
        config, raw = self.build(); raw['rope_scaling'] = {'rope_type': 'linear', 'factor': 2.}
        with self.assertRaises(ValueError): preserve_original_rope(config, raw, declared_context=8192)

if __name__ == '__main__': unittest.main()
