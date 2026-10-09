import copy
import types
import unittest

import torch

from subnet.fp32_gradient_accumulation import FP32GradientAccumulator, admit_capacity
from subnet.persistent_cpu_adamw import HYPERPARAMETERS
from subnet.task_normalized_training import accumulate_tasks
import test_persistent_training_policy as fixtures


class FP32GradientIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PersistentTrainingPolicyTests()
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def test_actual_pair_callback_keeps_744_gradients_in_fp32(self):
        parameter = torch.nn.Parameter(torch.tensor(0., dtype=torch.bfloat16))
        accumulator = FP32GradientAccumulator([('p', parameter)])
        tasks = [dict(pair_indices=list(range(4*i, 4*i+4)), task_sha256=f'{i:064x}')
                 for i in range(186)]
        records = accumulate_tasks(torch, lambda i: parameter.float(), [0.] * 744,
            tasks, list(range(186)), after_backward=accumulator.capture)
        self.assertEqual(len(records), 744)
        self.assertEqual(accumulator.microsteps, 744)
        self.assertIsNone(parameter.grad)
        accumulator.clip(1.)
        self.assertLess(abs(float(accumulator.gradients()['p']) + .05) / .05, .001)

    def test_optimizer_consumes_full_precision_without_roundtrip(self):
        optimizer = self.fixture.optimizer()
        parameter = self.fixture.parameters[0][1]
        reference = torch.nn.Parameter(parameter.detach().float())
        expected = torch.optim.AdamW([reference], lr=HYPERPARAMETERS['lr'], foreach=False,
            betas=tuple(HYPERPARAMETERS['betas']), eps=HYPERPARAMETERS['eps'],
            weight_decay=HYPERPARAMETERS['weight_decay'])
        gradient = torch.linspace(-.0499725, .031337, parameter.numel())
        self.assertFalse(torch.equal(gradient, gradient.bfloat16().float()))
        for _ in range(7):
            optimizer.step(gradients={'weight': gradient})
            reference.grad = gradient.clone()
            expected.step()
            self.assertIsNone(parameter.grad)
        for slot in ('exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(optimizer.rows['weight'][slot],
                expected.state[reference][slot], rtol=0, atol=0)
        torch.testing.assert_close(optimizer.rows['weight']['master'], reference.detach(), rtol=0, atol=0)
        self.assertEqual(optimizer.global_step, 7)
        self.assertTrue(torch.equal(parameter.detach(), reference.detach().bfloat16()))

    def test_rejected_inputs_leave_optimizer_state_untouched(self):
        optimizer = self.fixture.optimizer()
        parameter = self.fixture.parameters[0][1]
        original = copy.deepcopy(optimizer.rows)
        invalid = [dict(), {'other': torch.ones(12)}, {'weight': torch.ones(11)},
            {'weight': torch.ones(12, dtype=torch.bfloat16)},
            {'weight': torch.ones(12, requires_grad=True)},
            {'weight': torch.full((12,), float('nan'))},
            {'weight': torch.full((12,), float('inf'))}, {'weight': 'invalid'}]
        for gradients in invalid:
            with self.subTest(gradient_type=type(gradients.get('weight'))):
                with self.assertRaises(ValueError):
                    optimizer.step(gradients=gradients)
                self.assertEqual(optimizer.global_step, 0)
                for slot in ('master', 'exp_avg', 'exp_avg_sq'):
                    self.assertTrue(torch.equal(original['weight'][slot], optimizer.rows['weight'][slot]))
        parameter.grad = torch.ones_like(parameter)
        with self.assertRaisesRegex(ValueError, 'roundtrip'):
            optimizer.step(gradients={'weight': torch.ones(12)})
        self.assertEqual(optimizer.global_step, 0)

    def test_precision_transition_retains_restored_adam_lineage(self):
        optimizer = self.fixture.optimizer()
        self.fixture.steps(optimizer, 2)
        descriptor, _, store = self.fixture.publish(optimizer)
        restored, _ = self.fixture.restore(descriptor, store)
        successor = self.fixture.optimizer(input_checkpoint=self.fixture.successor, restored=restored)
        before_first_moment = successor.rows['weight']['exp_avg'].clone()
        successor.zero_grad()
        successor.step(gradients={'weight': torch.full((12,), .0499725)})
        self.assertEqual(successor.global_step, 3)
        self.assertEqual(successor.genesis_sha256, optimizer.genesis_sha256)
        self.assertEqual(successor.parent_state_sha256, fixtures.sha(descriptor))
        wanted = before_first_moment.lerp(torch.full((12,), .0499725), .1)
        torch.testing.assert_close(successor.rows['weight']['exp_avg'], wanted, rtol=0, atol=0)

    def test_capacity_admission_before_allocation(self):
        size = 8_000_000_000
        parameter = types.SimpleNamespace(device='cuda:0', is_cuda=True,
            numel=lambda: size, element_size=lambda: 2)
        framework = types.SimpleNamespace(cuda=types.SimpleNamespace(
            mem_get_info=lambda device: (100*1024**3, 140*1024**3)))
        result = admit_capacity(framework, [('p', parameter)])
        self.assertEqual(result['extra_buffer_bytes'], size * 4)
        self.assertTrue(result['admitted'])
        framework.cuda.mem_get_info = lambda device: (40*1024**3, 140*1024**3)
        with self.assertRaisesRegex(ValueError, 'insufficient GPU capacity'):
            admit_capacity(framework, [('p', parameter)])
        with self.assertRaisesRegex(ValueError, 'qualified CUDA'):
            admit_capacity(framework, self.fixture.parameters)


if __name__ == '__main__':
    unittest.main()
