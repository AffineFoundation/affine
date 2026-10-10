"""Task-normalized preference updates with persistent FP32 state.

Callers authenticate parent state/genesis and the manifest-selected admission
policy. An explicit committed-unaudited-training policy permits eligible pairs
without inference audit receipts; historical audited policies retain their
receipt requirements. GPU capacity and numerical contracts remain pinned by
the original job. Pair validity must not be inferred from legacy helper names.
"""
import hashlib
import math
import copy
import time
from pathlib import Path

from .covered_epoch_optimizer import distinct_verified_pairs, pair_identity
from .epoch_optimizer import preference_loss
from .persistent_cpu_adamw import POLICY, HYPERPARAMETERS, PersistentCPUAdamW, checkpoint_id
from .storage import canonical
from .fp32_gradient_accumulation import METHOD, FP32GradientAccumulator, admit_capacity


def task_groups(verified_pairs, steps, seed, *, required_pairs_per_task=None):
    if (type(steps) is not int or not 1 <= steps <= 32 or
            not verified_pairs or len(verified_pairs) > 65536):
        raise ValueError('task-normalized pair/update budget')
    checkpoint_id(seed)
    if required_pairs_per_task is not None and (type(required_pairs_per_task) is not int or not 2 <= required_pairs_per_task <= 64):
        raise ValueError('bounded explicit disjoint pairs per task')
    pairs = distinct_verified_pairs(verified_pairs)
    if required_pairs_per_task is not None:
        from .trajectory_identity import token_trace_sha256
        if len(pairs) != len(verified_pairs):
            raise ValueError('K2L2 duplicate pair forbidden')
        traces = {}
        for definition, positive, negative in pairs:
            key = definition['env_id'], positive['index']
            seen = traces.setdefault(key, set())
            for rollout in (positive, negative):
                identity = token_trace_sha256(rollout['turns'])
                if identity in seen:
                    raise ValueError('K2L2 shared rollout across disjoint pairs')
                seen.add(identity)
    identities = [pair_identity(p) for p in pairs]
    by_task = {}; hashes = {}
    for i, (definition, positive, negative) in enumerate(pairs):
        task_hash = positive.get('task_hash')
        checkpoint_id(task_hash)
        if negative.get('task_hash') != task_hash:
            raise ValueError('same-task positive/negative proof binding')
        key = definition['env_id'], positive['index']
        if key in hashes and hashes[key] != task_hash:
            raise ValueError('one task hash per environment/index')
        hashes[key] = task_hash
        by_task.setdefault(key, []).append(i)
    if required_pairs_per_task is not None and any(len(indices) != required_pairs_per_task for indices in by_task.values()):
        raise ValueError('K2L2 complete two-pair task required' if required_pairs_per_task==2 else 'complete manifest pair quota per task required')
    tasks = []
    for (env_id, index), indices in by_task.items():
        identity = dict(env_id=env_id, index=index, task_hash=hashes[env_id, index])
        tasks.append(dict(identity, task_sha256=hashlib.sha256(canonical(identity)).hexdigest(),
            pair_indices=sorted(indices, key=lambda i: identities[i])))
    tasks.sort(key=lambda row: hashlib.sha256(bytes.fromhex(seed) +
                                            bytes.fromhex(row['task_sha256'])).digest())
    if len(tasks) < steps:
        groups = [[i % len(tasks)] for i in range(steps)]
    else:
        size, remainder = divmod(len(tasks), steps); groups = []; offset = 0
        for step in range(steps):
            count = size + int(step < remainder)
            groups.append(list(range(offset, offset + count))); offset += count
    return pairs, tasks, groups, identities


def validate_positive_nll_weight(value):
    """The caller authenticates objective scope/range; this is only its scalar."""
    if type(value) not in (int, float) or not math.isfinite(value) or value not in (0, 1):
        raise ValueError('explicit zero or unit positive NLL coefficient')
    return value


def positive_nll_loss(torch, positive, negative, reference, beta=.1):
    """Unit positive NLL anchor, sharing the original two sequence forwards."""
    if type(beta) not in (int, float) or not math.isfinite(beta) or beta != .1:
        raise ValueError('unit positive NLL requires original preference beta')
    value = positive-negative
    preference = preference_loss(torch, value, reference, beta)
    nll = -positive
    loss = preference+nll
    if not all(bool(torch.isfinite(v)) for v in (positive, negative, value, preference, nll, loss)):
        raise ValueError('nonfinite positive NLL components')
    return loss, preference, nll, value


def capture_components(torch, components, pair_count, *, references=None, beta=.1):
    """Replace one existing reference/post pass; never add another model pass."""
    if type(pair_count) is not int or pair_count < 1 or (references is not None and len(references) != pair_count):
        raise ValueError('complete component reference population')
    rows = []
    with torch.no_grad():
        for i in range(pair_count):
            positive, negative, stated_margin = components(i)
            reference = float(stated_margin) if references is None else references[i]
            loss, preference, nll, value = positive_nll_loss(torch, positive, negative, reference, beta)
            if not torch.equal(value, stated_margin):
                raise ValueError('component/margin identity')
            rows.append(dict(pair_index=i, positive_mean_logprob=float(positive),
                negative_mean_logprob=float(negative), margin=float(value),
                preference_loss=float(preference), positive_nll=float(nll), loss=float(loss)))
    return rows


def weighted_component_summary(rows, tasks):
    """Mean pair within task, then mean task; no sequence-length reweighting."""
    if not tasks or [row['pair_index'] for row in rows] != list(range(len(rows))):
        raise ValueError('ordered complete component population')
    fields = ('positive_mean_logprob', 'negative_mean_logprob', 'margin',
              'preference_loss', 'positive_nll', 'loss')
    sums = {name: 0. for name in fields}; seen = []
    for task in tasks:
        indices = task['pair_indices']
        if not indices: raise ValueError('nonempty component task')
        for i in indices:
            if type(i) is not int or not 0 <= i < len(rows):
                raise ValueError('component pair index')
            seen.append(i); weight = 1/(len(tasks)*len(indices))
            for name in fields:
                value = rows[i][name]
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError('finite component summary')
                sums[name] += weight*value
    if sorted(seen) != list(range(len(rows))):
        raise ValueError('component task partition')
    return sums


def accumulate_tasks(torch, margin, references, tasks, indices, beta=.1, *, after_backward=None,
                     positive_nll_weight=0, components=None):
    """Mean pair loss within task, then mean task loss within update group."""
    validate_positive_nll_weight(positive_nll_weight)
    if positive_nll_weight and not callable(components):
        raise ValueError('unit positive NLL requires shared component forwards')
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('distinct nonempty task accumulation group')
    observations = []
    for task_index in indices:
        task = tasks[task_index]; pairs = task['pair_indices']
        if not pairs or len(set(pairs)) != len(pairs):
            raise ValueError('distinct nonempty per-task pair population')
        for i in pairs:
            if positive_nll_weight:
                positive, negative, stated_margin = components(i)
                loss, preference, nll, value = positive_nll_loss(torch, positive, negative, references[i], beta)
                if not torch.equal(value.detach(), stated_margin.detach()):
                    raise ValueError('component/margin identity')
            else:
                value = margin(i); loss = preference_loss(torch, value, references[i], beta)
            if not torch.isfinite(loss): raise ValueError('nonfinite task preference loss')
            weight = 1/(len(indices)*len(pairs))
            observation = dict(pair_index=i, task_index=task_index,
                task_sha256=task['task_sha256'], gradient_weight=weight,
                reference_margin=references[i], margin_before=float(value.detach()),
                loss=float(loss.detach()))
            if positive_nll_weight:
                observation.update(positive_mean_logprob=float(positive.detach()),
                    negative_mean_logprob=float(negative.detach()), preference_loss=float(preference.detach()),
                    positive_nll=float(nll.detach()), positive_nll_weight=1., beta=beta)
            observations.append(observation)
            (loss*weight).backward()
            if after_backward is not None:
                after_backward()
            del value, loss
            if positive_nll_weight:
                del positive, negative, stated_margin, preference, nll
    return observations


def train_epoch(runtime, verified_pairs, destination_root, *, input_checkpoint,
                epoch, seed, steps=3, approved_genesis=None,
                approved_genesis_sha256=None, restored_state=None,
                resource_admission, required_pairs_per_task=None,
                learning_rate_authorization=None, learning_rate_authority=None,
                job_id=None, positive_nll_weight=0):
    """Return BF16 export plus uncommitted persistent optimizer for publication.

    The backend must hash the BF16 export and export_state() with descriptor-last
    storage callbacks before it admits any next epoch. An unchanged BF16 export
    is allowed when FP32 state advances; this is not a learning-gain claim.
    """
    validate_positive_nll_weight(positive_nll_weight)
    import gc
    import torch
    from .protocol import harness_for
    checkpoint_id(input_checkpoint)
    if not isinstance(epoch, str) or not epoch or len(epoch) > 200:
        raise ValueError('exact epoch binding before training')
    pairs, tasks, groups, identities = task_groups(verified_pairs, steps, seed, required_pairs_per_task=required_pairs_per_task)
    model = runtime.model
    parameters = list(model.parameters())
    if (not parameters or any(not p.is_cuda or p.dtype != torch.bfloat16 for p in parameters) or
            any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules()) or
            getattr(model.config, 'attention_dropout', 0) != 0):
        raise ValueError('qualified dropout-free CUDA BF16 full-model profile required')
    destination = Path(destination_root)/'checkpoint-persistent-final'
    if destination.exists(): raise ValueError('refuse persistent checkpoint overwrite')
    phase_seconds={};phase_started=time.monotonic()
    optimizer = PersistentCPUAdamW(model.named_parameters(), input_checkpoint,
        approved_genesis=approved_genesis, approved_genesis_sha256=approved_genesis_sha256,
        restored=restored_state, resource_admission=resource_admission,
        learning_rate_authorization=learning_rate_authorization,
        learning_rate_authority=learning_rate_authority, epoch=epoch,
        job_id=job_id, steps=steps)
    phase_seconds['optimizer_initialization']=time.monotonic()-phase_started
    start_step = optimizer.global_step
    if positive_nll_weight and (optimizer.hyperparameters['lr'] != 5e-7 or
                               optimizer.hyperparameters['preference_beta'] != .1):
        raise ValueError('unit positive NLL keeps approved learning rate and beta')

    def sequence(rollout):
        total, tokens = 0, 0
        for turn in rollout['turns']:
            prompt, output = turn['prompt'], turn['output']
            logits = model(torch.tensor([prompt + output], device='cuda'),
                use_cache=False).logits[0, len(prompt)-1:len(prompt)+len(output)-1]
            total = total + torch.log_softmax(logits.float(), -1).gather(
                1, torch.tensor(output, device='cuda')[:, None]).sum()
            tokens += len(output)
        if not tokens: raise ValueError('empty verified training output')
        return total/tokens

    def margin(i):
        definition, positive, negative = pairs[i]
        runtime.configure(definition['spec'], harness_for(definition, positive['index']))
        return sequence(positive) - sequence(negative)

    def components(i):
        definition, positive, negative = pairs[i]
        runtime.configure(definition['spec'], harness_for(definition, positive['index']))
        positive_lp = sequence(positive)
        negative_lp = sequence(negative)
        return positive_lp, negative_lp, positive_lp-negative_lp

    model.eval()
    torch.cuda.synchronize();phase_started=time.monotonic()
    if positive_nll_weight:
        components_before = capture_components(torch, components, len(pairs))
        references = [row['margin'] for row in components_before]
    else:
        with torch.no_grad(): references = [float(margin(i)) for i in range(len(pairs))]
    torch.cuda.synchronize();phase_seconds['reference_forward']=time.monotonic()-phase_started
    if not all(math.isfinite(v) for v in references):
        raise ValueError('nonfinite immutable BF16 input reference')
    for parameter in parameters: parameter.requires_grad_(True)
    capacity = admit_capacity(torch, model.named_parameters())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train(); updates = []; seen = set(); torch.cuda.reset_peak_memory_stats()
    gradient_seconds=[];optimizer_seconds=[];accumulator=None
    try:
        for step, indices in enumerate(groups):
            optimizer.zero_grad()
            accumulator = FP32GradientAccumulator(model.named_parameters())
            torch.cuda.synchronize();phase_started=time.monotonic()
            observations = accumulate_tasks(torch, margin, references, tasks, indices,
                HYPERPARAMETERS['preference_beta'], after_backward=accumulator.capture,
                positive_nll_weight=positive_nll_weight, components=components if positive_nll_weight else None)
            if accumulator.microsteps != len(observations):
                raise ValueError('complete per-pair FP32 gradient capture required')
            norm = accumulator.clip(HYPERPARAMETERS['max_grad_norm'])
            torch.cuda.synchronize();gradient_seconds.append(time.monotonic()-phase_started)
            phase_started=time.monotonic()
            precision = optimizer.step(gradients=accumulator.gradients()); seen.update(indices)
            accumulator = None
            torch.cuda.synchronize();optimizer_seconds.append(time.monotonic()-phase_started)
            updates.append(dict(training_policy=POLICY, steps=1,
                epoch_optimizer_step=step+1, global_optimizer_step=optimizer.global_step,
                input_checkpoint=input_checkpoint, reference_scope='immutable-BF16-epoch-input',
                unique_tasks=len(tasks), unique_verified_pairs=len(pairs),
                gradient_tasks=len(indices), gradient_pairs=len(observations),
                cumulative_unique_gradient_tasks=len(seen),
                loss=sum(r['loss']*r['gradient_weight'] for r in observations),
                pairs=[dict(r, pair_sha256=identities[r['pair_index']]) for r in observations],
                gradient_norm_before_clip=float(norm),
                gradient_accumulation=METHOD, gradient_accumulation_dtype='torch.float32',
                gradient_capacity=capacity,
                full_model_finetune=True, gradient_tensors=len(parameters),
                hyperparameters=copy.deepcopy(optimizer.hyperparameters), precision=precision,
                task_weight_rule='mean-pair-within-task-then-mean-task-within-group',
                optimizer_lifecycle='persistent-across-epochs',
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            if positive_nll_weight:
                updates[-1].update(positive_nll_weight=1.,
                    preference_loss=sum(r['preference_loss']*r['gradient_weight'] for r in observations),
                    positive_nll=sum(r['positive_nll']*r['gradient_weight'] for r in observations),
                    positive_mean_logprob=sum(r['positive_mean_logprob']*r['gradient_weight'] for r in observations),
                    negative_mean_logprob=sum(r['negative_mean_logprob']*r['gradient_weight'] for r in observations))
        if seen != set(range(len(tasks))): raise ValueError('incomplete task gradient coverage')
        model.eval()
        torch.cuda.synchronize();phase_started=time.monotonic()
        if positive_nll_weight:
            components_after = capture_components(torch, components, len(pairs), references=references)
            after_margins = [row['margin'] for row in components_after]
        else:
            with torch.no_grad(): after_margins = [float(margin(i)) for i in range(len(pairs))]
        torch.cuda.synchronize();phase_seconds['post_update_forward']=time.monotonic()-phase_started
        if not all(math.isfinite(v) for v in after_margins):
            raise ValueError('nonfinite post-update training margin')
        phase_started=time.monotonic()
        destination.mkdir(mode=0o700)
        model.save_pretrained(destination, safe_serialization=True, max_shard_size='3.9GB')
        runtime.tokenizer.save_pretrained(destination)
        if any(path.stat().st_size > 4_000_000_000 for path in destination.iterdir() if path.is_file()):
            raise ValueError('actual BF16 export object cap')
        phase_seconds.update(checkpoint_save=time.monotonic()-phase_started,
            gradient_and_clip_by_step=gradient_seconds,CPU_optimizer_by_step=optimizer_seconds,
            GPU_timings_synchronized=True)
        diagnostics = dict(phase_seconds=phase_seconds,training_policy=POLICY, optimizer_steps=steps,
            global_optimizer_step_before=start_step, global_optimizer_step_after=optimizer.global_step,
            task_count=len(tasks), pair_count=len(pairs), updates=updates,
            master_state_updated=any(r['precision']['master_changed_elements'] for r in updates),
            inference_tensors_changed_during_updates=any(r['precision']['bf16_changed_elements'] for r in updates),
            training_pair_margin_before=references, training_pair_margin_after=after_margins,
            training_pair_margin_delta=[after-before for before, after in zip(references, after_margins)],
            heldout_gain_claimed=False, state_publication_required=True,
            complete=False, epoch=epoch, input_checkpoint=input_checkpoint,
            effective_hyperparameters=copy.deepcopy(optimizer.hyperparameters),
            learning_rate_authorization_sha256=(hashlib.sha256(canonical(learning_rate_authorization)).hexdigest()
                if learning_rate_authorization is not None else None))
        if positive_nll_weight:
            diagnostics['positive_nll_components'] = dict(
                version='unit-positive-nll-shared-forward-components-v1',
                positive_nll_weight=1., beta=.1, learning_rate=optimizer.hyperparameters['lr'],
                reference_scope='immutable-BF16-epoch-input',
                before=components_before, after=components_after,
                weighted_before=weighted_component_summary(components_before, tasks),
                weighted_after=weighted_component_summary(components_after, tasks),
                extra_model_forward_passes=0, tail_mask_applied=False)
        return destination, optimizer, diagnostics
    finally:
        accumulator = None
        optimizer.zero_grad(); model.eval(); model.gradient_checkpointing_disable()
        gc.collect(); torch.cuda.empty_cache()
