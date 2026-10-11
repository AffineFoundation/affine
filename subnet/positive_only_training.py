"""Prospective task-balanced positive SFT, behind a separately qualified release.

Admission still consumes complete manifest K/L batches. Only the positive
trajectories with their original terminal EOS enter this objective. The caller
authenticates the checkpoint, input admission, objective and optimizer lineage;
these diagnostics are bookkeeping, not a proof of execution or learning.
"""
import copy
import gc
import hashlib
import math
import time
from pathlib import Path

from .fp32_gradient_accumulation import METHOD, FP32GradientAccumulator, admit_capacity
from .persistent_cpu_adamw import POLICY, PersistentCPUAdamW, checkpoint_id
from .positive_only_objective import (VERSION as OBJECTIVE_VERSION,
    COMPLETION_POLICY, TASK_WEIGHT_RULE, configuration as objective_config)
from .storage import canonical
from .task_normalized_training import task_groups
from .trajectory_identity import token_trace_sha256

def fixed_task_minibatches(tasks, steps=8):
    """Partition an already frozen task order once, without replacement."""
    if type(steps) is not int or not 1 <= steps <= len(tasks):
        raise ValueError('nonempty bounded task minibatches required')
    identities = [r['task_sha256'] for r in tasks]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate task')
    all_rows = [r for task in tasks for r in task['positive_indices']]
    if not all_rows or any(not task['positive_indices'] for task in tasks):
        raise ValueError('every task requires a positive')
    if len(set(all_rows)) != len(all_rows):
        raise ValueError('a positive trajectory may appear only once')
    if any(type(r) is not int or r < 0 for r in all_rows):
        raise ValueError('nonnegative integer positive indices')
    size, remainder = divmod(len(tasks), steps)
    groups = []; offset = 0
    for step in range(steps):
        count = size + int(step < remainder)
        groups.append(list(range(offset, offset + count)))
        offset += count
    return groups


def accumulate_positive_tasks(torch, sequence, tasks, indices, *, after_backward):
    """Reviewed kernel: token mean, then positive mean, then task mean.

    `sequence(i)` scores generated tokens including the original terminal EOS.
    `after_backward` consumes and clears each microgradient into FP32.
    """
    if (not indices or len(set(indices)) != len(indices) or
            any(type(i) is not int or not 0 <= i < len(tasks) for i in indices)):
        raise ValueError('distinct nonempty task group')
    observations = []
    for task_index in indices:
        task = tasks[task_index]
        rows = task['positive_indices']
        if (not rows or len(set(rows)) != len(rows) or
                any(type(i) is not int or i < 0 for i in rows)):
            raise ValueError('distinct nonempty positive rows per task')
        weight = 1 / (len(indices) * len(rows))
        for row_index in rows:
            mean_logprob = sequence(row_index)
            if mean_logprob.ndim != 0 or not bool(torch.isfinite(mean_logprob)):
                raise ValueError('finite scalar generated-token log probability')
            loss = -mean_logprob
            observations.append(dict(positive_index=row_index, task_index=task_index,
                task_sha256=task['task_sha256'], weight=weight,
                positive_mean_logprob=float(mean_logprob.detach()), loss=float(loss.detach())))
            (loss * weight).backward()
            after_backward()
            del mean_logprob, loss
    if not math.isclose(sum(r['weight'] for r in observations), 1.0, abs_tol=1e-12):
        raise ValueError('task mean weights must sum to one')
    return observations


def _positive_completion(rollout, eos_token_id):
    turns = rollout.get('turns')
    if not isinstance(turns, list) or not turns:
        raise ValueError('nonempty admitted positive trajectory')
    reasons = []; generated_tokens = 0
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError('positive turn object')
        prompt, output = turn.get('prompt'), turn.get('output')
        if (not isinstance(prompt, list) or not prompt or
                not isinstance(output, list) or not output or
                any(type(t) is not int or t < 0 for t in prompt + output)):
            raise ValueError('nonempty integer prompt/output tokens')
        generated_tokens += len(output)
        if eos_token_id in output[:-1]:
            reasons.append(dict(turn_index=index, reason='eos-before-terminal'))
        if output[-1] != eos_token_id:
            reasons.append(dict(turn_index=index, reason='missing-terminal-eos'))
    return generated_tokens, reasons


def select_positive_tasks(pairs, steps, seed, eos_token_id, required_pairs_per_task):
    """Pure admission/selection for trainer and independent evidence checks.

    The supplied EOS must come from the authenticated checkpoint tokenizer, not
    submitted metadata. Missing EOS excludes an SFT response; it neither changes
    native grading nor establishes misconduct. No EOS is appended or invented.
    """
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise ValueError('pinned nonnegative integer EOS token')
    if type(steps) is not int or not 1 <= steps <= 32:
        raise ValueError('bounded positive-only update count')
    if type(required_pairs_per_task) is not int or not 2 <= required_pairs_per_task <= 64:
        raise ValueError('explicit complete manifest pair quota required')
    admitted, input_tasks, _, identities = task_groups(
        pairs, 1, seed, required_pairs_per_task=required_pairs_per_task)
    # Quota admission above validates task bindings and disjoint positive/negative
    # token traces before filtering. Negative tokens are never scored.
    input_task_by_pair = {i: task for task in input_tasks for i in task['pair_indices']}
    positives = []; retained = []; excluded = []; positive_by_pair = {}
    for pair_index, (definition, positive, _negative) in enumerate(admitted):
        task = input_task_by_pair[pair_index]
        tokens, reasons = _positive_completion(positive, eos_token_id)
        row = dict(source_pair_index=pair_index, pair_sha256=identities[pair_index],
            trajectory_sha256=token_trace_sha256(positive['turns']),
            env_id=task['env_id'], index=task['index'], task_hash=task['task_hash'],
            task_sha256=task['task_sha256'], generated_tokens=tokens,
            turn_count=len(positive['turns']))
        if reasons:
            excluded.append(dict(row, reason='not-terminal-eos-complete', turn_reasons=reasons))
        else:
            positive_index = len(positives)
            positive_by_pair[pair_index] = positive_index
            positives.append((definition, positive))
            retained.append(dict(row, positive_index=positive_index))
    tasks = []; excluded_tasks = []
    for task in input_tasks:
        identity = {key: task[key] for key in ('env_id', 'index', 'task_hash', 'task_sha256')}
        indices = [positive_by_pair[i] for i in task['pair_indices'] if i in positive_by_pair]
        if indices:
            tasks.append(dict(identity, positive_indices=indices))
        else:
            excluded_tasks.append(dict(identity, reason='no-terminal-eos-positives'))
    groups = fixed_task_minibatches(tasks, steps)
    selection = dict(version='terminal-eos-positive-selection-v1',
        completion_policy=COMPLETION_POLICY, eos_token_id=eos_token_id,
        required_pairs_per_task=required_pairs_per_task,
        input_pair_count=len(admitted), input_task_count=len(input_tasks),
        retained=retained, excluded=excluded, excluded_tasks=excluded_tasks,
        generated_tokens=sum(row['generated_tokens'] for row in retained),
        excluded_generated_tokens=sum(row['generated_tokens'] for row in excluded))
    return positives, tasks, groups, selection


def sequence_mean_logprob(torch, model, rollout, *, device):
    """Original next-token scorer: prompts condition but are never loss targets."""
    total, tokens = 0, 0
    for turn in rollout['turns']:
        prompt, output = turn['prompt'], turn['output']
        logits = model(torch.tensor([prompt + output], device=device),
            use_cache=False).logits[0, len(prompt)-1:len(prompt)+len(output)-1]
        total = total + torch.log_softmax(logits.float(), -1).gather(
            1, torch.tensor(output, device=device)[:, None]).sum()
        tokens += len(output)
    if not tokens:
        raise ValueError('empty admitted positive output')
    return total/tokens


def capture_positive_components(torch, sequence, positive_count):
    if type(positive_count) is not int or positive_count < 1:
        raise ValueError('nonempty positive component population')
    rows = []
    with torch.no_grad():
        for i in range(positive_count):
            value = sequence(i)
            if value.ndim != 0 or not bool(torch.isfinite(value)):
                raise ValueError('finite positive mean log probability')
            rows.append(dict(positive_index=i, positive_mean_logprob=float(value),
                             positive_nll=-float(value)))
    return rows


def weighted_positive_summary(rows, tasks):
    if not tasks or [row['positive_index'] for row in rows] != list(range(len(rows))):
        raise ValueError('complete ordered positive components')
    seen = []; total = 0.
    for task in tasks:
        indices = task['positive_indices']
        if not indices:
            raise ValueError('positive component task requires responses')
        for i in indices:
            if type(i) is not int or not 0 <= i < len(rows):
                raise ValueError('positive component index')
            row = rows[i]
            if (type(row['positive_nll']) not in (int, float) or
                    type(row['positive_mean_logprob']) not in (int, float) or
                    not math.isfinite(row['positive_nll']) or
                    row['positive_nll'] != -row['positive_mean_logprob']):
                raise ValueError('finite consistent positive components')
            total += row['positive_nll']/(len(tasks)*len(indices)); seen.append(i)
    if sorted(seen) != list(range(len(rows))):
        raise ValueError('exact positive task partition')
    return dict(positive_nll=total, positive_mean_logprob=-total)


def _admit_model_profile(torch, model):
    parameters = list(model.parameters())
    if (not parameters or any(not p.is_cuda or p.dtype != torch.bfloat16 for p in parameters) or
            len({p.device for p in parameters}) != 1 or
            any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules()) or
            getattr(model.config, 'attention_dropout', 0) != 0):
        raise ValueError('qualified dropout-free single-CUDA BF16 full-model profile required')
    return parameters


def train_epoch(runtime, verified_pairs, destination_root, *, input_checkpoint,
                epoch, seed, steps=3, approved_genesis=None,
                approved_genesis_sha256=None, restored_state=None,
                resource_admission, required_pairs_per_task=None,
                learning_rate_authorization=None, learning_rate_authority=None,
                job_id=None, objective):
    """Export one BF16 checkpoint; retain explicitly admitted optimizer lineage.

    As in the existing persistent trainer, publication/commit is the caller's
    responsibility. No fresh-genesis choice, LR override or release activation
    occurs here. The same authorized effective LR is used by every Adam step.
    """
    import torch
    from .protocol import harness_for
    objective = objective_config(objective)
    checkpoint_id(input_checkpoint)
    if not isinstance(epoch, str) or not epoch or len(epoch) > 200:
        raise ValueError('exact epoch binding before training')
    eos_token_id = runtime.tokenizer.eos_token_id
    positives, tasks, groups, selection = select_positive_tasks(
        verified_pairs, steps, seed, eos_token_id, required_pairs_per_task)
    model = runtime.model; parameters = _admit_model_profile(torch, model)
    destination = Path(destination_root)/'checkpoint-persistent-final'
    if destination.exists():
        raise ValueError('refuse persistent checkpoint overwrite')
    phase_seconds = {}; phase_started = time.monotonic()
    optimizer = PersistentCPUAdamW(model.named_parameters(), input_checkpoint,
        approved_genesis=approved_genesis, approved_genesis_sha256=approved_genesis_sha256,
        restored=restored_state, resource_admission=resource_admission,
        learning_rate_authorization=learning_rate_authorization,
        learning_rate_authority=learning_rate_authority, epoch=epoch,
        job_id=job_id, steps=steps)
    phase_seconds['optimizer_initialization'] = time.monotonic()-phase_started
    start_step = optimizer.global_step; accumulator = None

    def sequence(i):
        definition, positive = positives[i]
        runtime.configure(definition['spec'], harness_for(definition, positive['index']))
        return sequence_mean_logprob(torch, model, positive, device=parameters[0].device)

    try:
        model.eval(); torch.cuda.synchronize(); phase_started = time.monotonic()
        before = capture_positive_components(torch, sequence, len(positives))
        torch.cuda.synchronize(); phase_seconds['positive_before_forward'] = time.monotonic()-phase_started
        for parameter in parameters:
            parameter.requires_grad_(True)
        capacity = admit_capacity(torch, model.named_parameters())
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train(); updates = []; seen = set(); torch.cuda.reset_peak_memory_stats()
        gradient_seconds = []; optimizer_seconds = []
        for step, indices in enumerate(groups):
            optimizer.zero_grad()
            accumulator = FP32GradientAccumulator(model.named_parameters())
            torch.cuda.synchronize(); phase_started = time.monotonic()
            observations = accumulate_positive_tasks(torch, sequence, tasks, indices,
                after_backward=accumulator.capture)
            if accumulator.microsteps != len(observations):
                raise ValueError('complete per-positive FP32 gradient capture required')
            norm = accumulator.clip(optimizer.hyperparameters['max_grad_norm'])
            torch.cuda.synchronize(); gradient_seconds.append(time.monotonic()-phase_started)
            phase_started = time.monotonic()
            precision = optimizer.step(gradients=accumulator.gradients()); seen.update(indices)
            accumulator = None
            torch.cuda.synchronize(); optimizer_seconds.append(time.monotonic()-phase_started)
            attributed = []
            for row in observations:
                row = dict(row); weight = row.pop('weight')
                attributed.append(dict(selection['retained'][row['positive_index']], **row,
                                       gradient_weight=weight))
            loss = sum(row['loss']*row['gradient_weight'] for row in attributed)
            updates.append(dict(training_policy=POLICY, training_objective=copy.deepcopy(objective), steps=1,
                epoch_optimizer_step=step+1, global_optimizer_step=optimizer.global_step,
                input_checkpoint=input_checkpoint, unique_tasks=len(tasks),
                unique_positives=len(positives), gradient_tasks=len(indices),
                gradient_positives=len(attributed), cumulative_unique_gradient_tasks=len(seen),
                loss=loss, positive_nll=loss, positive_mean_logprob=-loss, positives=attributed,
                gradient_generated_tokens=sum(row['generated_tokens'] for row in attributed),
                gradient_norm_before_clip=float(norm), gradient_accumulation=METHOD,
                gradient_accumulation_dtype='torch.float32', gradient_capacity=capacity,
                full_model_finetune=True, gradient_tensors=len(parameters),
                hyperparameters=copy.deepcopy(optimizer.hyperparameters), precision=precision,
                task_weight_rule=TASK_WEIGHT_RULE, optimizer_lifecycle='persistent-across-epochs',
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        if seen != set(range(len(tasks))) or optimizer.global_step != start_step+steps:
            raise ValueError('complete once-through task coverage and optimizer steps required')
        model.eval(); torch.cuda.synchronize(); phase_started = time.monotonic()
        after = capture_positive_components(torch, sequence, len(positives))
        torch.cuda.synchronize(); phase_seconds['positive_after_forward'] = time.monotonic()-phase_started
        phase_started = time.monotonic(); destination.mkdir(mode=0o700)
        model.save_pretrained(destination, safe_serialization=True, max_shard_size='3.9GB')
        runtime.tokenizer.save_pretrained(destination)
        if any(path.stat().st_size > 4_000_000_000 for path in destination.iterdir() if path.is_file()):
            raise ValueError('actual BF16 export object cap')
        phase_seconds.update(checkpoint_save=time.monotonic()-phase_started,
            gradient_and_clip_by_step=gradient_seconds, CPU_optimizer_by_step=optimizer_seconds,
            GPU_timings_synchronized=True)
        diagnostics = dict(phase_seconds=phase_seconds, training_policy=POLICY,
            training_objective=objective, optimizer_steps=steps,
            global_optimizer_step_before=start_step, global_optimizer_step_after=optimizer.global_step,
            input_pair_count=selection['input_pair_count'], input_task_count=selection['input_task_count'],
            task_count=len(tasks), positive_count=len(positives), eos_token_id=eos_token_id,
            generated_tokens=selection['generated_tokens'], positive_selection=selection,
            tasks=tasks, task_groups=groups, updates=updates,
            coverage_seed=seed, coverage_schedule_sha256=hashlib.sha256(canonical(
                [[tasks[i]['task_sha256'] for i in group] for group in groups])).hexdigest(),
            master_state_updated=any(r['precision']['master_changed_elements'] for r in updates),
            inference_tensors_changed_during_updates=any(r['precision']['bf16_changed_elements'] for r in updates),
            positive_nll_components=dict(version='task-balanced-positive-only-components-v1',
                before=before, after=after, weighted_before=weighted_positive_summary(before, tasks),
                weighted_after=weighted_positive_summary(after, tasks),
                no_negative_forwards=True, completion_policy=COMPLETION_POLICY, eos_token_id=eos_token_id),
            heldout_gain_claimed=False, state_publication_required=True, complete=False,
            epoch=epoch, input_checkpoint=input_checkpoint,
            effective_hyperparameters=copy.deepcopy(optimizer.hyperparameters),
            learning_rate_authorization_sha256=(hashlib.sha256(canonical(learning_rate_authorization)).hexdigest()
                if learning_rate_authorization is not None else None))
        return destination, optimizer, diagnostics
    finally:
        accumulator = None
        optimizer.zero_grad(); model.eval(); model.gradient_checkpointing_disable()
        gc.collect(); torch.cuda.empty_cache()
