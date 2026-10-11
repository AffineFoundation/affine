"""Arithmetic and provenance checks; synthetic reports claim no GPU execution."""
import copy
import unittest

from subnet.positive_only_training import select_positive_tasks
from subnet.positive_only_training_evidence import validate_diagnostics
from subnet.positive_only_objective import VERSION, TASK_WEIGHT_RULE
from subnet.persistent_cpu_adamw import POLICY, HYPERPARAMETERS, sha
from subnet.fp32_gradient_accumulation import METHOD


def fixture():
    eos = 999
    pairs = []
    for task in range(3):
        for attempt in range(2):
            def rollout(label, output):
                return dict(classification=label, env_id='math', index=task,
                    task_hash=sha(dict(task=task)),
                    turns=[dict(prompt=[1, 2+task], output=output)])
            complete = task == 1 or (task == 0 and attempt == 0)
            positive = rollout('positive', [10+task*2+attempt, eos if complete else 23])
            negative = rollout('negative', [50+task*2+attempt, eos])
            pairs.append((dict(env_id='math', spec={}), positive, negative))
    objective = dict(version=VERSION, completion_policy='terminal-eos-v1',
                     task_weight_rule=TASK_WEIGHT_RULE)
    hp = dict(HYPERPARAMETERS, lr=1e-6)
    arguments = dict(objective=objective, input_checkpoint='a'*64, epoch='synthetic',
        steps=2, seed='b'*64, global_step_before=0,
        parameters=[dict(name='w', shape=[2], numel=2)], eos_token_id=eos,
        required_pairs_per_task=2, effective_hyperparameters=hp,
        learning_rate_authorization_sha256='c'*64)
    positives, tasks, groups, selection = select_positive_tasks(pairs, 2, 'b'*64, eos, 2)
    def snapshots(scale):
        rows = [dict(positive_index=i, positive_nll=scale*(i+1),
                     positive_mean_logprob=-scale*(i+1)) for i in range(len(positives))]
        mean = sum(sum(rows[i]['positive_nll'] for i in t['positive_indices'])/
                   len(t['positive_indices']) for t in tasks)/len(tasks)
        return rows, dict(positive_nll=mean, positive_mean_logprob=-mean)
    before, weighted_before = snapshots(1.0)
    after, weighted_after = snapshots(0.9)
    updates = []; seen = set()
    for step, group in enumerate(groups):
        seen.update(group); observations = []
        for task_index in group:
            task = tasks[task_index]
            for i in task['positive_indices']:
                observations.append(dict(selection['retained'][i], task_index=task_index,
                    gradient_weight=1/(len(group)*len(task['positive_indices'])),
                    positive_mean_logprob=-(i+1.0), loss=i+1.0))
        loss = sum(row['loss']*row['gradient_weight'] for row in observations)
        precision = dict(optimizer_step=step+1, master_dtype='torch.float32',
            optimizer_state_dtype='torch.float32', inference_dtype='torch.bfloat16',
            effective_hyperparameters=hp, learning_rate_authorization_sha256='c'*64,
            parameters=[dict(name='w', elements=2, master_changed_elements=2,
                bf16_changed_elements=0, master_delta_l2=1e-6, master_delta_max_abs=1e-6)],
            master_changed_elements=2, bf16_changed_elements=0)
        updates.append(dict(training_policy=POLICY, training_objective=objective,
            steps=1, epoch_optimizer_step=step+1, global_optimizer_step=step+1,
            input_checkpoint='a'*64, unique_tasks=len(tasks), unique_positives=len(positives),
            gradient_tasks=len(group), gradient_positives=len(observations),
            cumulative_unique_gradient_tasks=len(seen), loss=loss, positive_nll=loss,
            positive_mean_logprob=-loss, positives=observations,
            gradient_generated_tokens=sum(r['generated_tokens'] for r in observations),
            gradient_norm_before_clip=0.2, gradient_accumulation=METHOD,
            gradient_accumulation_dtype='torch.float32', full_model_finetune=True,
            gradient_tensors=1, hyperparameters=hp, precision=precision,
            task_weight_rule=TASK_WEIGHT_RULE, optimizer_lifecycle='persistent-across-epochs'))
    diagnostic = dict(training_policy=POLICY, training_objective=objective, optimizer_steps=2,
        global_optimizer_step_before=0, global_optimizer_step_after=2,
        input_pair_count=6, input_task_count=3, task_count=2, positive_count=3,
        eos_token_id=eos, generated_tokens=selection['generated_tokens'],
        positive_selection=selection, tasks=tasks, task_groups=groups, coverage_seed='b'*64,
        coverage_schedule_sha256=sha([[tasks[i]['task_sha256'] for i in g] for g in groups]),
        master_state_updated=True, inference_tensors_changed_during_updates=False,
        heldout_gain_claimed=False, state_publication_required=True, complete=False,
        epoch='synthetic', input_checkpoint='a'*64, effective_hyperparameters=hp,
        learning_rate_authorization_sha256='c'*64,
        positive_nll_components=dict(version='task-balanced-positive-only-components-v1',
            before=before, after=after, weighted_before=weighted_before, weighted_after=weighted_after,
            no_negative_forwards=True, completion_policy='terminal-eos-v1', eos_token_id=eos))
    return diagnostic, updates, pairs, arguments


class PositiveOnlyEvidenceTests(unittest.TestCase):
    def test_accepts_exact_unequal_positive_counts_and_records_exclusions(self):
        d, u, p, args = fixture()
        result = validate_diagnostics(d, u, p, **args)
        self.assertEqual(result['distinct_positives'], 3)
        self.assertEqual(result['distinct_tasks'], 2)
        self.assertEqual(result['excluded_positives'], 3)
        self.assertEqual(result['excluded_tasks'], 1)
        self.assertFalse(result['historical_gpu_execution_proven'])

    def test_rejects_report_mutations(self):
        mutations = {
            'checkpoint': lambda d,u: d.update(input_checkpoint='f'*64),
            'epoch': lambda d,u: d.update(epoch='another'),
            'eos': lambda d,u: d.update(eos_token_id=23),
            'unreported-exclusion': lambda d,u: d['positive_selection']['excluded'].pop(),
            'renamed-negative': lambda d,u: u[0].update(gradient_pairs=1),
            'missing-positive': lambda d,u: u[0]['positives'].pop(),
            'double-count': lambda d,u: u[0]['positives'].append(copy.deepcopy(u[0]['positives'][0])),
            'gradient-weight': lambda d,u: u[0]['positives'][0].update(gradient_weight=0.1),
            'wrong-source': lambda d,u: u[0]['positives'][0].update(trajectory_sha256='d'*64),
            'snapshot-mean': lambda d,u: d['positive_nll_components']['weighted_before'].update(positive_nll=42),
            'nan': lambda d,u: u[0].update(loss=float('nan')),
            'negative-nll': lambda d,u: u[0]['positives'][0].update(loss=-1, positive_mean_logprob=1),
            'lr': lambda d,u: u[0].update(hyperparameters=dict(HYPERPARAMETERS, lr=9e-6)),
            'optimizer-counter': lambda d,u: u[0]['precision'].update(optimizer_step=9),
            'fp16-master': lambda d,u: u[0]['precision'].update(master_dtype='torch.float16'),
            'parameter-identity': lambda d,u: u[0]['precision']['parameters'][0].update(name='other'),
            'changed-count': lambda d,u: u[0]['precision']['parameters'][0].update(master_changed_elements=3),
            'reset-lineage': lambda d,u: d.update(global_optimizer_step_before=5),
            'fabricated-gain': lambda d,u: d.update(heldout_gain_claimed=True),
            'boolean-count': lambda d,u: u[0].update(steps=True),
        }
        for name, mutate in mutations.items():
            d, u, p, args = fixture(); mutate(d, u)
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_diagnostics(d, u, p, **args)

    def test_report_cannot_choose_its_own_eos_or_learning_rate(self):
        for field, value in [('eos_token_id', 23),
                ('effective_hyperparameters', dict(HYPERPARAMETERS, lr=2e-6)),
                ('learning_rate_authorization_sha256', 'f'*64)]:
            d, u, p, args = fixture(); args[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_diagnostics(d, u, p, **args)

    def test_input_changes_cannot_be_hidden_in_original_report(self):
        d, u, p, args = fixture()
        p[0][1]['turns'][0]['output'][0] = 80
        with self.assertRaises(ValueError):
            validate_diagnostics(d, u, p, **args)


if __name__ == '__main__':
    unittest.main()
