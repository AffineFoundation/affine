"""Pure bookkeeping checks for a separately authorized positive-only update.

The caller must authenticate the job, checkpoint/tokenizer EOS, input admissions,
optimizer lineage and effective hyperparameters before calling this module.
Reported losses are not proofs of physical computation or generalization.
This module is not wired into any production release yet.
"""
import math

from .fp32_gradient_accumulation import METHOD
from .persistent_cpu_adamw import POLICY, sha
from .positive_only_objective import COMPLETION_POLICY, TASK_WEIGHT_RULE, configuration
from .positive_only_training import select_positive_tasks


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _equal(actual, expected):
    return _number(actual) and _number(expected) and math.isclose(
        actual, expected, rel_tol=1e-6, abs_tol=1e-7)


def _exact(actual, expected, message):
    # Canonical equality also rejects booleans used in place of integer counts.
    if sha(actual) != sha(expected):
        raise ValueError(message)


def _components(row, *, nll_key='positive_nll'):
    nll, logprob = row.get(nll_key), row.get('positive_mean_logprob')
    if (not _number(nll) or nll < 0 or not _number(logprob) or logprob > 0 or
            not _equal(nll, -logprob)):
        raise ValueError('finite nonnegative positive NLL and log-probability arithmetic')
    return nll


def _precision(value, inventory, step, effective_hyperparameters, grant_sha256):
    if not isinstance(value, dict):
        raise ValueError('positive SFT optimizer precision evidence')
    for key, expected in dict(optimizer_step=step, master_dtype='torch.float32',
            optimizer_state_dtype='torch.float32', inference_dtype='torch.bfloat16',
            effective_hyperparameters=effective_hyperparameters,
            learning_rate_authorization_sha256=grant_sha256).items():
        _exact(value.get(key), expected, 'positive SFT precision/counter/rate binding')
    rows = value.get('parameters')
    if not isinstance(rows, list) or len(rows) != len(inventory):
        raise ValueError('complete positive SFT parameter inventory')
    for row, approved in zip(rows, inventory):
        if not isinstance(row, dict):
            raise ValueError('parameter precision object')
        _exact(row.get('name'), approved['name'], 'parameter precision name')
        _exact(row.get('elements'), approved['numel'], 'parameter precision size')
        for key in ('master_changed_elements', 'bf16_changed_elements'):
            if type(row.get(key)) is not int or not 0 <= row[key] <= approved['numel']:
                raise ValueError('bounded parameter changed-element counts')
        for key in ('master_delta_l2', 'master_delta_max_abs'):
            if not _number(row.get(key)) or row[key] < 0:
                raise ValueError('finite parameter movement evidence')
    for key in ('master_changed_elements', 'bf16_changed_elements'):
        _exact(value.get(key), sum(row[key] for row in rows),
               'complete aggregate parameter changed-element counts')
    return value['master_changed_elements'] > 0, value['bf16_changed_elements'] > 0


def validate_diagnostics(diagnostics, updates, admitted_pairs, *, objective,
        input_checkpoint, epoch, steps, seed, global_step_before, parameters,
        eos_token_id, required_pairs_per_task, effective_hyperparameters,
        learning_rate_authorization_sha256):
    """Reconstruct exact selection, exclusions, task weights and update coverage.

Every keyword denotes independently authenticated expectations, especially EOS,
LR, lineage and the complete positive/negative admission population. Never fill
these arguments from the report being checked.
    """
    objective = configuration(objective)
    if (not isinstance(diagnostics, dict) or 'updates' in diagnostics or
            not isinstance(updates, list) or len(updates) != steps or
            type(global_step_before) is not int or global_step_before < 0 or
            not isinstance(parameters, list) or not parameters):
        raise ValueError('positive SFT diagnostic/update structure')
    positives, tasks, groups, selection = select_positive_tasks(
        admitted_pairs, steps, seed, eos_token_id, required_pairs_per_task)
    expected = dict(training_policy=POLICY, training_objective=objective,
        optimizer_steps=steps, global_optimizer_step_before=global_step_before,
        global_optimizer_step_after=global_step_before+steps,
        input_pair_count=selection['input_pair_count'],
        input_task_count=selection['input_task_count'], task_count=len(tasks),
        positive_count=len(positives), eos_token_id=eos_token_id,
        generated_tokens=selection['generated_tokens'], positive_selection=selection,
        tasks=tasks, task_groups=groups, coverage_seed=seed,
        coverage_schedule_sha256=sha([[tasks[i]['task_sha256'] for i in g] for g in groups]),
        heldout_gain_claimed=False, state_publication_required=True, complete=False,
        epoch=epoch, input_checkpoint=input_checkpoint,
        effective_hyperparameters=effective_hyperparameters,
        learning_rate_authorization_sha256=learning_rate_authorization_sha256)
    for key, value in expected.items():
        _exact(diagnostics.get(key), value, 'positive SFT selection/job/lineage binding: '+key)
    # Catch misleading legacy objective fields rather than silently ignoring them.
    legacy = {'pair_count', 'gradient_pairs', 'pairs', 'reference_scope',
        'reference_margin', 'margin_before', 'margin', 'preference_loss',
        'negative_mean_logprob', 'positive_nll_weight', 'positive_nll_components_legacy',
        'training_pair_margin_before', 'training_pair_margin_after', 'training_pair_margin_delta'}
    locations = [diagnostics, *updates]
    for update in updates:
        if not isinstance(update, dict) or not isinstance(update.get('positives'), list):
            raise ValueError('positive SFT update population')
        locations.extend(update['positives'])
    if any(not isinstance(row, dict) or legacy.intersection(row) for row in locations):
        raise ValueError('positive-only report cannot claim preference/negative forwards')

    components = diagnostics.get('positive_nll_components')
    keys = {'version', 'before', 'after', 'weighted_before', 'weighted_after',
            'no_negative_forwards', 'completion_policy', 'eos_token_id'}
    if (not isinstance(components, dict) or set(components) != keys or
            components['version'] != 'task-balanced-positive-only-components-v1' or
            components['no_negative_forwards'] is not True or
            components['completion_policy'] != COMPLETION_POLICY):
        raise ValueError('explicit positive-only before/after components')
    _exact(components['eos_token_id'], eos_token_id, 'independently pinned component EOS')
    for when in ('before', 'after'):
        rows = components[when]
        if not isinstance(rows, list) or len(rows) != len(positives):
            raise ValueError('complete before/after positive population')
        for i, row in enumerate(rows):
            if (not isinstance(row, dict) or
                    set(row) != {'positive_index', 'positive_mean_logprob', 'positive_nll'}):
                raise ValueError('exact positive snapshot components')
            _exact(row['positive_index'], i, 'ordered positive snapshot index')
            _components(row)
        mean = sum(sum(rows[i]['positive_nll'] for i in task['positive_indices']) /
                   len(task['positive_indices']) for task in tasks) / len(tasks)
        summary = components['weighted_'+when]
        if (not isinstance(summary, dict) or set(summary) != {'positive_nll', 'positive_mean_logprob'}
                or not _equal(_components(summary), mean)):
            raise ValueError('equal-task positive NLL snapshot arithmetic')

    seen = set(); master_changed = False; bf16_changed = False
    for step, (update, group) in enumerate(zip(updates, groups)):
        seen.update(group)
        population = []
        for task_index in group:
            task = tasks[task_index]
            for i in task['positive_indices']:
                population.append(dict(selection['retained'][i], task_index=task_index,
                    gradient_weight=1/(len(group)*len(task['positive_indices']))))
        expected_update = dict(training_policy=POLICY, training_objective=objective,
            steps=1, epoch_optimizer_step=step+1, global_optimizer_step=global_step_before+step+1,
            input_checkpoint=input_checkpoint, unique_tasks=len(tasks),
            unique_positives=len(positives), gradient_tasks=len(group),
            gradient_positives=len(population), cumulative_unique_gradient_tasks=len(seen),
            gradient_generated_tokens=sum(row['generated_tokens'] for row in population),
            gradient_accumulation=METHOD, gradient_accumulation_dtype='torch.float32',
            full_model_finetune=True, gradient_tensors=len(parameters),
            hyperparameters=effective_hyperparameters, task_weight_rule=TASK_WEIGHT_RULE,
            optimizer_lifecycle='persistent-across-epochs')
        for key, value in expected_update.items():
            _exact(update.get(key), value, 'positive SFT update attribution: '+key)
        if (not _number(update.get('gradient_norm_before_clip')) or update['gradient_norm_before_clip'] < 0
                or len(update['positives']) != len(population)):
            raise ValueError('finite gradient and complete positive update population')
        total = 0.
        for row, wanted in zip(update['positives'], population):
            if set(row) != set(wanted) | {'positive_mean_logprob', 'loss'}:
                raise ValueError('exact positive update observation fields')
            for key, value in wanted.items():
                _exact(row.get(key), value, 'positive SFT source/weight attribution: '+key)
            total += _components(row, nll_key='loss')*wanted['gradient_weight']
        if (not _equal(update.get('loss'), total) or
                not _equal(_components(update), total)):
            raise ValueError('task-balanced positive update loss arithmetic')
        master, bf16 = _precision(update.get('precision'), parameters,
            global_step_before+step+1, effective_hyperparameters,
            learning_rate_authorization_sha256)
        master_changed |= master; bf16_changed |= bf16
    if seen != set(range(len(tasks))):
        raise ValueError('complete once-through positive task coverage')
    _exact(diagnostics.get('master_state_updated'), master_changed, 'master update summary')
    _exact(diagnostics.get('inference_tensors_changed_during_updates'), bf16_changed,
           'BF16 update summary')
    return dict(update_count=len(updates), distinct_tasks=len(tasks),
        distinct_positives=len(positives), excluded_positives=len(selection['excluded']),
        excluded_tasks=len(selection['excluded_tasks']),
        global_step_before=global_step_before, global_step_after=global_step_before+steps,
        master_state_updated=master_changed, heldout_gain_claimed=False,
        historical_gpu_execution_proven=False)
