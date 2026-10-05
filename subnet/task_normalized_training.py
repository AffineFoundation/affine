"""Prospective task-normalized preference updates with persistent FP32 state.

Not selected by existing workers. Callers must independently verify every pair,
authenticate parent state/genesis, and qualify actual GPU forward/backward
capacity before this implementation receives a production job.
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


def task_groups(verified_pairs, steps, seed):
    if (type(steps) is not int or not 1 <= steps <= 32 or
            not verified_pairs or len(verified_pairs) > 65536):
        raise ValueError('task-normalized pair/update budget')
    checkpoint_id(seed)
    pairs = distinct_verified_pairs(verified_pairs)
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


def accumulate_tasks(torch, margin, references, tasks, indices, beta=.1):
    """Mean pair loss within task, then mean task loss within update group."""
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('distinct nonempty task accumulation group')
    observations = []
    for task_index in indices:
        task = tasks[task_index]; pairs = task['pair_indices']
        if not pairs or len(set(pairs)) != len(pairs):
            raise ValueError('distinct nonempty per-task pair population')
        for i in pairs:
            value = margin(i); loss = preference_loss(torch, value, references[i], beta)
            if not torch.isfinite(loss): raise ValueError('nonfinite task preference loss')
            weight = 1/(len(indices)*len(pairs))
            observations.append(dict(pair_index=i, task_index=task_index,
                task_sha256=task['task_sha256'], gradient_weight=weight,
                reference_margin=references[i], margin_before=float(value.detach()),
                loss=float(loss.detach())))
            (loss*weight).backward()
            del value, loss
    return observations


def train_epoch(runtime, verified_pairs, destination_root, *, input_checkpoint,
                epoch, seed, steps=3, approved_genesis=None,
                approved_genesis_sha256=None, restored_state=None,
                resource_admission):
    """Return BF16 export plus uncommitted persistent optimizer for publication.

    The backend must hash the BF16 export and export_state() with descriptor-last
    storage callbacks before it admits any next epoch. An unchanged BF16 export
    is allowed when FP32 state advances; this is not a learning-gain claim.
    """
    import gc
    import torch
    from .protocol import harness_for
    checkpoint_id(input_checkpoint)
    if not isinstance(epoch, str) or not epoch or len(epoch) > 200:
        raise ValueError('exact epoch binding before training')
    pairs, tasks, groups, identities = task_groups(verified_pairs, steps, seed)
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
        restored=restored_state, resource_admission=resource_admission)
    phase_seconds['optimizer_initialization']=time.monotonic()-phase_started
    start_step = optimizer.global_step

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

    model.eval()
    torch.cuda.synchronize();phase_started=time.monotonic()
    with torch.no_grad(): references = [float(margin(i)) for i in range(len(pairs))]
    torch.cuda.synchronize();phase_seconds['reference_forward']=time.monotonic()-phase_started
    if not all(math.isfinite(v) for v in references):
        raise ValueError('nonfinite immutable BF16 input reference')
    for parameter in parameters: parameter.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train(); updates = []; seen = set(); torch.cuda.reset_peak_memory_stats()
    gradient_seconds=[];optimizer_seconds=[]
    try:
        for step, indices in enumerate(groups):
            optimizer.zero_grad()
            torch.cuda.synchronize();phase_started=time.monotonic()
            observations = accumulate_tasks(torch, margin, references, tasks, indices,
                                            HYPERPARAMETERS['preference_beta'])
            if any(p.grad is None for p in parameters):
                raise ValueError('full-model task-normalized gradient coverage')
            norm = torch.nn.utils.clip_grad_norm_(parameters,
                HYPERPARAMETERS['max_grad_norm'], error_if_nonfinite=True)
            torch.cuda.synchronize();gradient_seconds.append(time.monotonic()-phase_started)
            phase_started=time.monotonic()
            precision = optimizer.step(); seen.update(indices)
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
                full_model_finetune=True, gradient_tensors=len(parameters),
                hyperparameters=copy.deepcopy(HYPERPARAMETERS), precision=precision,
                task_weight_rule='mean-pair-within-task-then-mean-task-within-group',
                optimizer_lifecycle='persistent-across-epochs',
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        if seen != set(range(len(tasks))): raise ValueError('incomplete task gradient coverage')
        model.eval()
        torch.cuda.synchronize();phase_started=time.monotonic()
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
            complete=False, epoch=epoch, input_checkpoint=input_checkpoint)
        return destination, optimizer, diagnostics
    finally:
        optimizer.zero_grad(); model.eval(); model.gradient_checkpointing_disable()
        gc.collect(); torch.cuda.empty_cache()
