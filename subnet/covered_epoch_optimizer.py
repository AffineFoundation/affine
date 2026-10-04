"""Prospective full-coverage preference training; not selected by live workers.

The caller must independently verify every pair against the input checkpoint.
Groups accumulate one pair's gradients at a time, keeping activation memory
bounded. Only the completed final checkpoint is exported.
"""
import hashlib
import math
from pathlib import Path

from .epoch_optimizer import preference_loss
from .storage import canonical

POLICY = 'bf16-full-adamw-covered-fixed-reference-v3'


def coverage_schedule(pairs, steps, seed):
    """Bind ordering to frozen pair contents and a post-freeze challenge seed."""
    if (type(steps) is not int or not 1 <= steps <= 32 or
            not pairs or len(pairs) > 65536):
        raise ValueError('covered epoch pair/step budget')
    if (not isinstance(seed, str) or len(seed) != 64 or
            any(c not in '0123456789abcdef' for c in seed)):
        raise ValueError('post-freeze coverage seed')
    identities = []
    for definition, positive, negative in pairs:
        if (positive.get('classification') != 'positive' or
                negative.get('classification') != 'negative' or
                type(positive.get('index')) is not int or positive['index'] < 0 or
                type(negative.get('index')) is not int or
                positive['index'] != negative['index'] or
                positive.get('env_id') != definition['env_id'] or
                negative.get('env_id') != definition['env_id']):
            raise ValueError('covered training pair binding')
        identities.append(hashlib.sha256(canonical(dict(
            env_id=definition['env_id'], index=positive['index'],
            positive=positive, negative=negative))).hexdigest())
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate covered training pair')
    ordered = sorted(range(len(pairs)), key=lambda i: hashlib.sha256(
        bytes.fromhex(seed) + bytes.fromhex(identities[i])).digest())
    if len(ordered) < steps:
        groups = [[ordered[i % len(ordered)]] for i in range(steps)]
    else:
        size, remainder = divmod(len(ordered), steps)
        groups = []; offset = 0
        for step in range(steps):
            count = size + int(step < remainder)
            groups.append(ordered[offset:offset + count]); offset += count
    return groups, identities


def accumulate_group(torch, margin, references, indices, beta=.1):
    """Backward the group mean sequentially, never retain all pair graphs."""
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('nonempty distinct accumulation group')
    observations = []
    for i in indices:
        value = margin(i)
        loss = preference_loss(torch, value, references[i], beta)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite covered preference loss')
        observations.append(dict(pair_index=i, reference_margin=references[i],
                                 margin_before=float(value.detach()),
                                 loss=float(loss.detach())))
        (loss / len(indices)).backward()
        del value, loss
    return observations


def train_epoch(runtime, verified_pairs, destination_root, *, seed, steps=3):
    import gc
    import torch
    from .backend_jobs import pair_attribution
    from .protocol import harness_for

    groups, identities = coverage_schedule(verified_pairs, steps, seed)
    destination = Path(destination_root) / 'checkpoint-covered-final'
    if destination.exists():
        raise ValueError('refuse covered checkpoint overwrite')
    model = runtime.model
    parameters = list(model.parameters())
    if not parameters or any(not p.is_cuda or p.dtype != torch.bfloat16 for p in parameters):
        raise ValueError('approved CUDA BF16 full-model profile required')
    if (any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules()) or
            getattr(model.config, 'attention_dropout', 0) != 0):
        raise ValueError('dropout-free covered reference')
    count = sum(p.numel() for p in parameters)
    free, _ = torch.cuda.mem_get_info()
    required = count * 6 + 3 * 1024**3
    if free < required:
        raise ValueError('persistent covered optimizer GPU reserve')

    def sequence(rollout):
        total, tokens = 0, 0
        for turn in rollout['turns']:
            prompt, output = turn['prompt'], turn['output']
            logits = model(torch.tensor([prompt + output], device='cuda'),
                           use_cache=False).logits[0, len(prompt)-1:len(prompt)+len(output)-1]
            total = total + torch.log_softmax(logits.float(), -1).gather(
                1, torch.tensor(output, device='cuda')[:, None]).sum()
            tokens += len(output)
        if not tokens:
            raise ValueError('empty approved training trajectory')
        return total / tokens

    def margin(i):
        definition, pos, neg = verified_pairs[i]
        runtime.configure(definition['spec'], harness_for(definition, pos['index']))
        return sequence(pos) - sequence(neg)

    model.eval()
    with torch.no_grad():
        references = [float(margin(i)) for i in range(len(verified_pairs))]
    if not all(math.isfinite(v) for v in references):
        raise ValueError('nonfinite immutable covered reference')
    for parameter in parameters:
        parameter.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, foreach=False)
    updates = []; seen = set(); torch.cuda.reset_peak_memory_stats()
    try:
        for step, indices in enumerate(groups):
            optimizer.zero_grad(set_to_none=True)
            observations = accumulate_group(torch, margin, references, indices)
            gradient_tensors = sum(p.grad is not None for p in parameters)
            if gradient_tensors != len(parameters):
                raise ValueError('full covered optimizer gradient coverage')
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
            optimizer.step(); seen.update(indices)
            state_steps = sorted({int(row['step'].item()) for row in optimizer.state.values()})
            if len(optimizer.state) != len(parameters) or state_steps != [step + 1]:
                raise ValueError('persistent covered optimizer state coverage')
            attribution = []
            for observation in observations:
                i = observation['pair_index']
                definition, pos, neg = verified_pairs[i]
                attribution.append(dict(observation, pair_sha256=identities[i],
                                        **pair_attribution(definition, pos, neg, step)))
            updates.append(dict(
                steps=1, optimizer_step=step+1, training_policy=POLICY,
                objective='group-mean fixed-input-reference sequence preference',
                losses=[sum(r['loss'] for r in observations) / len(observations)],
                input_pairs=len(verified_pairs), gradient_pairs=len(indices),
                cumulative_unique_gradient_pairs=len(seen), pairs=attribution,
                coverage_seed=seed, coverage_schedule_sha256=hashlib.sha256(canonical(
                    [[identities[i] for i in group] for group in groups])).hexdigest(),
                reference_scope='immutable-epoch-input-before-all-updates',
                full_model_finetune=True, trainable_parameters=count, learning_rate=1e-5,
                beta=.1, parameter_dtype='torch.bfloat16', gradient_checkpointing=True,
                gradient_tensors=gradient_tensors, total_parameter_tensors=len(parameters),
                gradient_norm_before_clip=float(norm), optimizer_state_steps=state_steps,
                optimizer_state_dtypes=sorted({str(v.dtype) for row in optimizer.state.values()
                    for k, v in row.items() if k != 'step' and hasattr(v, 'dtype')}),
                optimizer_lifecycle='one-AdamW-instance-all-epoch-steps',
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                gpu_free_before_bytes=free, gpu_required_additional_bytes=required,
                checkpoint_export_policy='one-final-after-complete-coverage'))
        if seen != set(range(len(verified_pairs))):
            raise ValueError('incomplete gradient pair coverage')
        destination.mkdir(parents=True)
        model.save_pretrained(destination, safe_serialization=True, max_shard_size='4GB')
        runtime.tokenizer.save_pretrained(destination)
        return destination, updates
    finally:
        optimizer.zero_grad(set_to_none=True); del optimizer
        model.eval(); model.gradient_checkpointing_disable()
        gc.collect(); torch.cuda.empty_cache()
