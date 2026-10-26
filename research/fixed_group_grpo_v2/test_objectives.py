import unittest
from dataclasses import replace
import torch
from .core import Policy, Stamp, Rollout, Group, validate_group, group_eligibility, draw_and_record
from .objectives import Objective, group_advantages, grpo_trajectory_loss, preference_coefficients
from .scoring import selected_logprobs_from_logits


def fixture():
    p = Policy(temperature=.7, support_size=4, eos_token_id=3, max_output_tokens=3, max_context_tokens=8)
    s = Stamp('fixed', 'a'*64, 0, 'b'*64, p)
    rows = tuple(Rollout(i, (i%2, 3), (-.7, -.5), 'eos', 'correct' if i < 4 else 'wrong') for i in range(8))
    return Group(s, 'task', (0, 1), rows)


class ObjectiveTests(unittest.TestCase):
    def test_complete_group_and_duplicate_text(self):
        g = fixture()
        self.assertTrue(group_eligibility(g, 8)['eligible'])
        self.assertEqual(g.rollouts[0].output, g.rollouts[2].output)
        validate_group(g, 8)

    def test_missing_duplicate_or_out_of_range_nonce_rejected(self):
        g = fixture()
        for rows in (g.rollouts[:-1], g.rollouts[:-1]+(replace(g.rollouts[-1], attempt=6),),
                     g.rollouts[:-1]+(replace(g.rollouts[-1], attempt=8),), tuple(reversed(g.rollouts))):
            with self.assertRaises(ValueError): validate_group(replace(g, rollouts=rows), 8)

    def test_all_positive_or_negative_filter(self):
        for outcome in ('correct', 'wrong'):
            g = fixture()
            self.assertFalse(group_eligibility(replace(g, rollouts=tuple(replace(r, outcome=outcome) for r in g.rollouts)), 8)['eligible'])

    def test_cap_is_unresolved_not_failure(self):
        g = fixture()
        cap = replace(g.rollouts[-1], output=(0, 1, 2), behavior_logprobs=(-1.,)*3, stop='budget', outcome='unresolved')
        validate_group(replace(g, rollouts=g.rollouts[:-1]+(cap,)), 8)
        with self.assertRaises(ValueError): validate_group(replace(g, rollouts=g.rollouts[:-1]+(replace(cap, outcome='wrong'),)), 8)

    def test_group_population_statistics(self):
        a, stats = group_advantages(fixture(), group_size=8)
        self.assertAlmostEqual(stats['mean'], .5)
        self.assertAlmostEqual(stats['population_std'], .5)
        self.assertAlmostEqual(a[0], .5/.5001)
        self.assertAlmostEqual(a[-1], -.5/.5001)

    def test_initial_ratio_and_gradient_direction(self):
        for advantage in (-1., 1.):
            x = torch.tensor([-1., -2.], dtype=torch.float64, requires_grad=True)
            loss, d = grpo_trajectory_loss(x, x.detach().tolist(), x.detach().tolist(), advantage=advantage,
                objective=Objective(), task_count=1, group_size=8)
            self.assertEqual(d['mean_ratio'], 1.)
            loss.backward()
            self.assertTrue(bool((x.grad * advantage < 0).all()))

    def test_clipping_both_advantage_signs(self):
        for delta, advantage in ((.4, 1.), (-.4, -1.)):
            x = torch.tensor([-2.+delta], dtype=torch.float64, requires_grad=True)
            loss, _ = grpo_trajectory_loss(x, [-2.], [-2.], advantage=advantage,
                objective=Objective(kl_beta=0.), task_count=1, group_size=8)
            loss.backward()
            self.assertEqual(float(x.grad), 0.)

    def test_kl_is_differentiable_not_detached_reward(self):
        x = torch.tensor([-1.], dtype=torch.float64, requires_grad=True)
        loss, d = grpo_trajectory_loss(x, [-1.], [-2.], advantage=0.,
            objective=Objective(kl_beta=.01), task_count=1, group_size=8)
        loss.backward()
        self.assertGreater(d['mean_kl'], 0.)
        self.assertAlmostEqual(float(x.grad), .01*(1-torch.exp(torch.tensor(-1.)).item())/8, places=8)

    def test_unresolved_masks_every_loss_term(self):
        x = torch.tensor([-1.], dtype=torch.float64, requires_grad=True)
        loss, _ = grpo_trajectory_loss(x, [-2.], [-3.], advantage=-1.,
            objective=Objective(), task_count=1, group_size=8, active=False)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(float(x.grad), 0.)

    def test_probability_temperature_support_and_eos(self):
        g = fixture(); p = g.stamp.policy
        logits = torch.tensor([[.1,.2,.3,.4,1e6]]*4, requires_grad=True)
        lp = selected_logprobs_from_logits(logits, g.prompt, (0, 3), p, chunk_tokens=1)
        expected = torch.log_softmax(logits[0,:4]/.7, -1)[torch.tensor([0,3])]
        self.assertTrue(torch.allclose(lp, expected.double(), atol=1e-7))
        lp.sum().backward()
        self.assertEqual(float(logits.grad[:,4].abs().sum()), 0.)
        self.assertEqual(float(logits.grad[0].abs().sum()), 0.)
        self.assertEqual(float(logits.grad[3].abs().sum()), 0.)
        token, value = draw_and_record(logits[0].detach(), .99, p)
        self.assertEqual(token, 3)
        self.assertAlmostEqual(value, float(expected[1].detach()), places=6)

    def test_factorized_control_matches_full_autograd(self):
        g = fixture(); means = [-1., -2., -3., -4., -2., -3., -4., -5.]
        old = [-2.]*8
        coefficients, detail = preference_coefficients(g, means, old, group_size=8, beta=.1, task_count=3)
        x = torch.tensor(means, dtype=torch.float64, requires_grad=True)
        full = torch.stack([-torch.nn.functional.logsigmoid(.1*(x[i]-x[j])) for i in range(4) for j in range(4,8)]).mean()/3
        full.backward()
        self.assertTrue(torch.allclose(torch.tensor(coefficients), x.grad.float(), atol=1e-8))
        self.assertEqual(detail['comparison_count'], 16)


if __name__ == '__main__': unittest.main()
