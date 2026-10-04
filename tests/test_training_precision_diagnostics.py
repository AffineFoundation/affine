"""Numerical learning controls, without loading weights or using a GPU.

These demonstrate a risk in the current BF16 AdamW policy. They do not estimate
the fraction of real checkpoint values affected and do not authorize a policy
change. A changed checkpoint hash is weaker than an effective learning update.
"""
import unittest

import torch

from subnet.covered_epoch_optimizer import accumulate_group


class TrainingPrecisionDiagnostics(unittest.TestCase):
    def test_bf16_current_policy_can_discard_three_nonzero_updates(self):
        initial = torch.tensor([1.0, .1, .02, .01, .005], dtype=torch.bfloat16)
        low = torch.nn.Parameter(initial.clone())
        precise = torch.nn.Parameter(initial.float())
        before = precise.detach().clone()
        low_optimizer = torch.optim.AdamW([low], lr=1e-5, foreach=False)
        precise_optimizer = torch.optim.AdamW([precise], lr=1e-5, foreach=False)
        for _ in range(3):
            low.grad = torch.ones_like(low)
            precise.grad = torch.ones_like(precise)
            low_optimizer.step()
            precise_optimizer.step()
        self.assertTrue(torch.equal(low.detach(), initial))
        self.assertTrue(bool(torch.all(precise.detach() < before)))
        self.assertEqual(low_optimizer.state[low]['exp_avg'].dtype, torch.bfloat16)
        self.assertEqual(precise_optimizer.state[precise]['exp_avg'].dtype, torch.float32)

    def test_reloading_rounded_export_loses_cross_epoch_progress(self):
        initial = torch.tensor([.02], dtype=torch.bfloat16)
        exported = initial.clone()
        for _ in range(4):
            ephemeral = torch.nn.Parameter(exported.float())
            optimizer = torch.optim.AdamW([ephemeral], lr=1e-5, foreach=False)
            for _ in range(3):
                ephemeral.grad = torch.ones_like(ephemeral)
                optimizer.step()
            exported = ephemeral.detach().to(torch.bfloat16)
        self.assertTrue(torch.equal(exported, initial))

        master = torch.nn.Parameter(initial.float())
        optimizer = torch.optim.AdamW([master], lr=1e-5, foreach=False)
        for _ in range(12):
            master.grad = torch.ones_like(master)
            optimizer.step()
        self.assertLess(float(master.detach().to(torch.bfloat16)), float(initial))

    def test_covered_preference_gradient_has_correct_direction_in_fp32(self):
        # Toy chosen/rejected token log probabilities. This checks the sign of
        # the actual group-mean objective, independently of BF16 rounding.
        values = torch.nn.Parameter(torch.tensor([[-.8, -.5], [-1.1, -.7]]))
        references = [float(v[0] - v[1]) for v in values.detach()]
        before = values.detach().clone()
        optimizer = torch.optim.SGD([values], lr=.1)
        accumulate_group(torch, lambda i: values[i, 0] - values[i, 1],
                         references, [0, 1])
        optimizer.step()
        self.assertTrue(bool(torch.all(values.detach()[:, 0] > before[:, 0])))
        self.assertTrue(bool(torch.all(values.detach()[:, 1] < before[:, 1])))


if __name__ == '__main__':
    unittest.main()
