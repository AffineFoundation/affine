import copy
import unittest
import torch

from subnet.covered_epoch_optimizer import coverage_schedule, accumulate_group
from subnet.epoch_optimizer import preference_loss


def pairs(n):
    return [(dict(env_id='math'), dict(env_id='math', index=i,
            classification='positive', turns=[{'output': [i+1]}]),
            dict(env_id='math', index=i, classification='negative',
                 turns=[{'output': [i+2]}])) for i in range(n)]


class CoveredEpochTests(unittest.TestCase):
    def test_large_population_all_contribute_without_repetition(self):
        groups, _ = coverage_schedule(pairs(175), 3, 'ab'*32)
        self.assertEqual([len(g) for g in groups], [59, 58, 58])
        self.assertEqual(sorted(i for g in groups for i in g), list(range(175)))

    def test_content_order_independent_of_submission_iteration(self):
        original = pairs(12)
        g, identities = coverage_schedule(original, 3, 'cd'*32)
        h, reversed_ids = coverage_schedule(original[::-1], 3, 'cd'*32)
        self.assertEqual([[identities[i] for i in x] for x in g],
                         [[reversed_ids[i] for i in x] for x in h])

    def test_accumulation_matches_dense_group_mean_gradient(self):
        accumulated = torch.nn.Parameter(torch.tensor([.2, -.1, .8, .5]))
        dense = torch.nn.Parameter(accumulated.detach().clone())
        refs = [.1, -.2, .7, .4]
        observations = accumulate_group(torch, lambda i: accumulated[i], refs, [3, 0, 2, 1])
        torch.stack([preference_loss(torch, dense[i], refs[i]) for i in range(4)]).mean().backward()
        torch.testing.assert_close(accumulated.grad, dense.grad)
        self.assertEqual(len(observations), 4)

    def test_pairs_after_first_three_receive_real_optimizer_updates(self):
        population = pairs(9); groups, _ = coverage_schedule(population, 3, 'ef'*32)
        values = torch.nn.Parameter(torch.zeros(9)); refs = [0.]*9
        optimizer = torch.optim.AdamW([values], lr=.1, weight_decay=0)
        seen = set()
        for group in groups:
            optimizer.zero_grad(set_to_none=True)
            accumulate_group(torch, lambda i: values[i], refs, group)
            seen.update(torch.nonzero(values.grad).flatten().tolist())
            optimizer.step()
        self.assertEqual(seen, set(range(9)))
        self.assertTrue(bool(torch.all(values.detach() > 0)))
        self.assertEqual(int(optimizer.state[values]['step']), 3)

    def test_small_population_has_no_empty_groups(self):
        groups, _ = coverage_schedule(pairs(2), 3, '12'*32)
        self.assertEqual([len(g) for g in groups], [1, 1, 1])
        self.assertEqual({i for g in groups for i in g}, {0, 1})

    def test_refuses_bad_bindings_duplicates_and_seed(self):
        for field, value in [('classification', 'negative'), ('index', True), ('env_id', 'other')]:
            bad = copy.deepcopy(pairs(1)); bad[0][1][field] = value
            with self.assertRaises(ValueError): coverage_schedule(bad, 1, '12'*32)
        with self.assertRaises(ValueError): coverage_schedule(pairs(1)*2, 1, '12'*32)
        for seed in ('bad', 'gg'*32, None):
            with self.assertRaises(ValueError): coverage_schedule(pairs(1), 1, seed)
        for steps in (0, True, 33):
            with self.assertRaises(ValueError): coverage_schedule(pairs(1), steps, '12'*32)

    def test_nonfinite_gradient_loss_refused(self):
        value = torch.nn.Parameter(torch.tensor(float('nan')))
        with self.assertRaises(ValueError): accumulate_group(torch, lambda i: value, [0.], [0])


if __name__ == '__main__':
    unittest.main()
