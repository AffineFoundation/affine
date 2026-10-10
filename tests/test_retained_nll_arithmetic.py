"""Portable CPU arithmetic controls; no model download or GPU qualification."""
import math
import unittest

import torch

from subnet import task_normalized_training as training
from subnet.fp32_gradient_accumulation import FP32GradientAccumulator
import test_learning_rate_transition as lr_fixtures


TASKS = [dict(task_sha256='a'*64, pair_indices=[0]),
         dict(task_sha256='b'*64, pair_indices=[2, 1, 3])]
TOKENS = [(0, 1), (1, 2), (2, 0), (0, 2)]
REFERENCES = [.12, -.37, .6, -.18]


def components_for(parameter, calls=None):
    def components(index):
        if calls is not None:
            calls.append(index)
        logprobs = parameter.float().log_softmax(0)
        positive, negative = (logprobs[i] for i in TOKENS[index])
        return positive, negative, positive-negative
    return components


def accumulated(parameter, coefficient=None):
    accumulator = FP32GradientAccumulator([('weight', parameter)])
    components = components_for(parameter)
    options = {} if coefficient is None else dict(positive_nll_weight=coefficient)
    def forbidden(index):
        raise AssertionError('zero objective must not request component telemetry')
    options['components'] = components if coefficient else forbidden
    rows = training.accumulate_tasks(
        torch, lambda i: components(i)[2], REFERENCES, TASKS, [0, 1],
        after_backward=accumulator.capture, **options)
    return rows, accumulator.buffers['weight'].clone(), accumulator


class RetainedNLLArithmetic(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_analytic_loss_and_positive_negative_gradients(self):
        positive = torch.tensor(-2., dtype=torch.float64, requires_grad=True)
        negative = torch.tensor(-3., dtype=torch.float64, requires_grad=True)
        loss, preference, nll, margin = training.positive_nll_loss(
            torch, positive, negative, 1.)
        self.assertAlmostEqual(loss.item(), math.log(2)+2, places=14)
        self.assertAlmostEqual(preference.item(), math.log(2), places=14)
        self.assertEqual((nll.item(), margin.item()), (2., 1.))
        loss.backward()
        self.assertAlmostEqual(positive.grad.item(), -1.05, places=14)
        self.assertAlmostEqual(negative.grad.item(), .05, places=14)

    def test_task_weighted_gradient_matches_independent_closed_form(self):
        for coefficient in (0, 1):
            with self.subTest(coefficient=coefficient):
                parameter = torch.nn.Parameter(torch.tensor([.5, .25, -.75]))
                rows, actual, _ = accumulated(parameter, coefficient)
                probabilities = parameter.detach().double().softmax(0)
                expected = torch.zeros(3, dtype=torch.float64)
                weights = [.5, 1/6, 1/6, 1/6]
                for i, (positive, negative) in enumerate(TOKENS):
                    margin = float(parameter.detach()[positive]-parameter.detach()[negative])
                    slope = -.1/(1+math.exp(.1*(margin-REFERENCES[i])))
                    derivative = coefficient*probabilities.clone()
                    derivative[positive] += slope-coefficient
                    derivative[negative] -= slope
                    expected += weights[i]*derivative
                torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=2e-8)
                for task in range(2):
                    self.assertAlmostEqual(sum(r['gradient_weight'] for r in rows
                                               if r['task_index']==task), .5)

    def test_default_and_explicit_zero_match_for_fp32_and_bf16(self):
        for dtype in (torch.float32, torch.bfloat16):
            def run(coefficient):
                return accumulated(torch.nn.Parameter(torch.tensor([.5, .25, -.75], dtype=dtype)), coefficient)
            original = run(None)
            for coefficient in (0, 0.):
                rows, gradient, _ = run(coefficient)
                self.assertEqual(rows, original[0])
                self.assertTrue(torch.equal(gradient, original[1]))
                self.assertFalse(any('positive_nll' in row for row in rows))

    def test_component_telemetry_reuses_three_existing_pair_passes(self):
        parameter = torch.nn.Parameter(torch.tensor([.5, .25, -.75]))
        calls = []
        components = components_for(parameter, calls)
        before = training.capture_components(torch, components, 4)
        references = [r['margin'] for r in before]
        accumulator = FP32GradientAccumulator([('weight', parameter)])
        def forbidden(index):
            raise AssertionError('unit objective must share the component forward')
        rows = training.accumulate_tasks(
            torch, forbidden, references, TASKS, [0, 1],
            positive_nll_weight=1, components=components,
            after_backward=accumulator.capture)
        after = training.capture_components(torch, components, 4, references=references)
        self.assertEqual(calls, [0, 1, 2, 3, 0, 2, 1, 3, 0, 1, 2, 3])
        self.assertEqual(accumulator.microsteps, 4)
        self.assertEqual(before, after)
        summary = training.weighted_component_summary(before, TASKS)
        for field in ('loss', 'positive_nll', 'preference_loss'):
            self.assertAlmostEqual(summary[field], sum(r[field]*r['gradient_weight'] for r in rows))

    def test_nonzero_adam_continuation_matches_torch_without_reset(self):
        fixture = lr_fixtures.LRTransitionTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        optimizer = fixture.continuation(fixture.grant(rate=5e-7, steps=2), steps=2)
        row = optimizer.rows['weight']
        self.assertEqual(optimizer.global_step, 4)
        self.assertTrue(torch.count_nonzero(row['exp_avg']) > 0)
        self.assertTrue(torch.count_nonzero(row['exp_avg_sq']) > 0)
        identities = {k: id(row[k]) for k in ('master', 'exp_avg', 'exp_avg_sq')}
        reference = torch.nn.Parameter(row['master'].clone())
        expected = torch.optim.AdamW([reference], lr=5e-7, betas=(.9, .999), eps=1e-8,
                                    weight_decay=.01, foreach=False)
        expected.state[reference] = dict(step=torch.tensor(4.), exp_avg=row['exp_avg'].clone(),
                                         exp_avg_sq=row['exp_avg_sq'].clone())
        for step, coefficient in enumerate((0, 1), 5):
            optimizer.zero_grad()
            parameter = optimizer.parameters[0][1]
            _, _, accumulator = accumulated(parameter, coefficient)
            accumulator.clip(1.)
            gradient = accumulator.gradients()['weight']
            reference.grad = gradient.clone()
            expected.step()
            optimizer.step(gradients={'weight': gradient})
            self.assertEqual((optimizer.global_step, row['step']), (step, step))
            targets = dict(master=reference.detach(), exp_avg=expected.state[reference]['exp_avg'],
                           exp_avg_sq=expected.state[reference]['exp_avg_sq'])
            for name, target in targets.items():
                self.assertEqual(id(row[name]), identities[name])
                torch.testing.assert_close(row[name], target, rtol=0, atol=0)
            self.assertTrue(torch.equal(parameter.detach(), row['master'].to(torch.bfloat16)))

    def test_bad_coefficient_or_components_refused_before_forward(self):
        for coefficient in (True, False, None, '1', -.1, .5, 2, float('nan'), float('inf')):
            with self.subTest(coefficient=repr(coefficient)), self.assertRaises(ValueError):
                training.train_epoch(None, [], None, input_checkpoint=None, epoch=None,
                                     seed=None, resource_admission=None,
                                     positive_nll_weight=coefficient)
        with self.assertRaisesRegex(ValueError, 'shared component'):
            training.accumulate_tasks(torch, None, REFERENCES, TASKS, [0, 1], positive_nll_weight=1)

    def test_mismatched_margin_and_changed_beta_refused(self):
        with self.assertRaisesRegex(ValueError, 'component/margin'):
            training.capture_components(torch, lambda i: (torch.tensor(-2.), torch.tensor(-3.), torch.tensor(0.)), 1)
        for beta in (True, .2, float('nan')):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                training.positive_nll_loss(torch, torch.tensor(-2.), torch.tensor(-3.), 1., beta)


if __name__ == '__main__':
    unittest.main()
