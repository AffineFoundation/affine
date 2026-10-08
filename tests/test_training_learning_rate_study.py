import copy
import math
import unittest

from ops.training_learning_rate_study import research_optimizer, FP32GradientAccumulator
from subnet.persistent_cpu_adamw import HYPERPARAMETERS, PersistentCPUAdamW


class ResearchLearningRateTests(unittest.TestCase):
    def test_large_microbatch_accumulation_does_not_saturate_bf16(self):
        import torch
        ordinary = torch.nn.Parameter(torch.tensor(1., dtype=torch.bfloat16))
        precise = torch.nn.Parameter(ordinary.detach().clone())
        increment = torch.tensor(1 / 900, dtype=torch.bfloat16)
        with FP32GradientAccumulator([precise]) as accumulator:
            for unused in range(900):
                (ordinary * increment).backward()
                (precise * increment).backward()
            accumulator.flush()
        expected = float(increment) * 900
        self.assertLess(abs(float(precise.grad) - expected), .004)
        self.assertGreater(abs(float(ordinary.grad) - expected), .25)

    def test_missing_gradient_and_nonfinite_accumulation_fail(self):
        import torch
        a = torch.nn.Parameter(torch.tensor(1., dtype=torch.bfloat16))
        b = torch.nn.Parameter(a.detach().clone())
        with FP32GradientAccumulator([a, b]) as accumulator:
            a.backward()
            with self.assertRaisesRegex(ValueError, 'incomplete'): accumulator.flush()
        with FP32GradientAccumulator([a]) as accumulator:
            (a * float('inf')).backward()
            with self.assertRaisesRegex(ValueError, 'nonfinite'): accumulator.flush()

    def test_parent_moments_and_bias_correction_match_adamw(self):
        import torch
        for lr in (1e-5, 1e-6, 5e-7):
            with self.subTest(lr=lr):
                cls = research_optimizer(PersistentCPUAdamW, lr)
                optimizer = cls.__new__(cls)
                parameter = torch.nn.Parameter(torch.tensor([.2, -.3], dtype=torch.bfloat16))
                parameter.grad = torch.tensor([.03, -.02], dtype=torch.bfloat16)
                master = torch.tensor([.2001, -.3001], dtype=torch.float32)
                moment = torch.tensor([.004, -.006], dtype=torch.float32)
                second = torch.tensor([.0002, .0003], dtype=torch.float32)
                expected = torch.nn.Parameter(master.clone())
                reference = torch.optim.AdamW([expected], lr=lr, betas=(.9, .999), eps=1e-8, weight_decay=.01, foreach=False)
                reference.state[expected] = dict(step=torch.tensor(33.), exp_avg=moment.clone(), exp_avg_sq=second.clone())
                expected.grad = parameter.grad.float().clone()
                reference.step()
                optimizer.parameters = [('p', parameter)]
                optimizer.rows = {'p': dict(master=master.clone(), exp_avg=moment.clone(), exp_avg_sq=second.clone(), step=33)}
                optimizer.global_step = 33
                optimizer.hyperparameters = copy.deepcopy(HYPERPARAMETERS)
                optimizer.research_learning_rate = lr
                optimizer._step_impl()
                self.assertEqual(optimizer.global_step, 34)
                torch.testing.assert_close(optimizer.rows['p']['master'], expected.detach(), rtol=0, atol=3e-8)
                torch.testing.assert_close(parameter.detach(), expected.detach().to(torch.bfloat16), rtol=0, atol=0)
                self.assertEqual(optimizer.hyperparameters, HYPERPARAMETERS)

    def test_invalid_learning_rate_and_changed_parent_hypers_rejected(self):
        for lr in (0, -1, math.inf, math.nan, 1e-4, True):
            with self.assertRaises(ValueError): research_optimizer(PersistentCPUAdamW, lr)
        cls = research_optimizer(PersistentCPUAdamW, 5e-7)
        optimizer = cls.__new__(cls)
        optimizer.hyperparameters = dict(HYPERPARAMETERS, lr=1e-6)
        optimizer.research_learning_rate = 5e-7
        with self.assertRaisesRegex(ValueError, 'immutable optimizer'):
            optimizer._step_impl()


if __name__ == '__main__': unittest.main()
