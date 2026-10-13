"""One complete optimizer minibatch using the pinned MiMo 9B loss.

OLD is recomputed by the trusted training engine before any update, matching
upstream v1 with rollout_correction.bypass_mode=false. It is not the miner's
claimed LP, the top-k/top-p distribution, or our older experimental denominator.
"""
from dataclasses import asdict
import math
import time
import numpy as np
import torch

from .config import TrainerConfig, MASK_UNRESOLVED
from .data import InputBinding, TaskGroup, digest
from .optimizer import FP32Accumulator
from .upstream import reference, policy_loss


def _versions(model):
    return tuple((name, id(p), p._version) for name, p in model.named_parameters())


def _layout(groups, config):
    """Derive global prompt weights BEFORE partitioning into microbatches."""
    rows = []
    for group in groups:
        rewards = torch.tensor([[float(r.outcome == 'correct')] for r in group.trajectories])
        advantages, _ = reference().compute_grpo_outcome_advantage(rewards,
            torch.ones_like(rewards), np.array([group.task_id] * config.group_size),
            norm_adv_by_std_in_grpo=False)
        for rollout, advantage in zip(group.trajectories, advantages[:, 0], strict=True):
            active = not (rollout.outcome == 'unresolved' and config.cap_policy == MASK_UNRESOLVED)
            rows.append(dict(group=group, rollout=rollout, advantage=float(advantage), active=active))
    longest = max(len(row['rollout'].tokens) for row in rows)
    mask = torch.zeros((len(rows), longest), dtype=torch.bool)
    for index, row in enumerate(rows):
        if row['active']:
            mask[index, :len(row['rollout'].tokens)] = True
    weights = reference().compute_prompt_loss_weights(mask, [row['group'].task_id for row in rows])
    for row, weight in zip(rows, weights, strict=True):
        row['prompt_weight'] = float(weight)
    return rows


def train_update(runtime, optimizer, groups, binding: InputBinding, *,
                 config=TrainerConfig(), guard=lambda: None,
                 record=lambda row: None, microbatch_rollouts=1):
    """Advance the caller's leased actor once, after accumulating whole groups.

    `runtime` supplies model, checkpoint_id, vocab_size, eos_token_ids,
    max_context_tokens and score(prompt, output, gradients=bool). The caller
    authenticates the contract and native admission receipts before this API.
    `optimizer` is a persistent CPUAdamW with matching named actor parameters.
    Publication is deliberately a separate transaction stage.
    """
    config.validate()
    binding.validate()
    groups = tuple(groups)
    if type(microbatch_rollouts) is not int or microbatch_rollouts < 1:
        raise ValueError('positive rollout microbatch size')
    if len(groups) > config.max_tasks or len({g.task_id for g in groups}) != len(groups):
        raise ValueError('distinct complete tasks within optimizer batch limit')
    if runtime.checkpoint_id != binding.checkpoint or optimizer.input_checkpoint != binding.checkpoint:
        raise ValueError('trusted runtime/optimizer must match input checkpoint')
    if optimizer.global_step != binding.optimizer_step or optimizer.config != config or optimizer.failed:
        raise ValueError('original healthy optimizer step and resolved config')
    expected = tuple((name, id(p)) for name, p in runtime.model.named_parameters() if p.requires_grad)
    if expected != tuple((name, id(p)) for name, p in optimizer.parameters):
        raise ValueError('optimizer must own exact trainable actor parameters')
    for group in groups:
        if not isinstance(group, TaskGroup):
            raise ValueError('native trusted TaskGroup required')
        group.validate(config.group_size, vocab_size=runtime.vocab_size, eos_token_ids=set(runtime.eos_token_ids))
        if any(len(group.prompt) + len(r.tokens) > runtime.max_context_tokens for r in group.trajectories):
            raise ValueError('complete rollout exceeds qualified total context')
    guard()
    base_report = dict(version='affine-mimo-grpo-update-v1', config=config.document(), input=asdict(binding),
        admitted_groups=[dict(task_id=g.task_id, admission_sha256=g.admission_sha256) for g in groups],
        group_count=len(groups), rollout_count=len(groups) * config.group_size,
        input_checkpoint=binding.checkpoint, optimizer_step_before=optimizer.global_step)
    if not groups:
        return dict(base_report, status='no-update', reason='no-eligible-complete-groups',
                    optimizer_step=optimizer.global_step)
    started = time.monotonic()
    runtime.model.eval()
    versions = _versions(runtime.model)
    rows = _layout(groups, config)
    preparation_started = time.monotonic()
    # Complete all OLD computations against one unchanged checkpoint, before
    # gradients and before any optimizer update. No miner LP field is accepted.
    for row in rows:
        guard()
        if not row['active']:
            row['old'] = None
            continue
        with torch.no_grad():
            old = runtime.score(row['group'].prompt, row['rollout'].tokens, gradients=False)
        if old.shape != (len(row['rollout'].tokens),) or not torch.isfinite(old).all():
            raise ValueError('trusted OLD requires finite selected-token probabilities')
        row['old'] = old.detach().to(device='cpu', dtype=torch.float32).clone()
        del old
    preparation_seconds = time.monotonic() - preparation_started
    if _versions(runtime.model) != versions:
        raise ValueError('input actor changed during OLD preparation')
    optimizer.zero_grad()
    accumulator = FP32Accumulator(runtime.model.named_parameters())
    losses, details = [], []
    pending = []
    maximum_drift = 0.
    try:
        for row in rows:
            guard()
            rollout, group = row['rollout'], row['group']
            if not row['active']:
                record(dict(stage='masked-unresolved', task_id=group.task_id, nonce=rollout.nonce,
                    tokens=len(rollout.tokens), reward=0., advantage=row['advantage']))
                continue
            if _versions(runtime.model) != versions:
                raise ValueError('actor mutated before the single optimizer boundary')
            with torch.enable_grad():
                current = runtime.score(group.prompt, rollout.tokens, gradients=True)
            if current.shape != row['old'].shape or not torch.isfinite(current).all():
                raise ValueError('finite differentiable current token probabilities required')
            old = row['old'].to(device=current.device, dtype=current.dtype)
            drift = float((current.detach() - old).abs().max())
            maximum_drift = max(maximum_drift, drift)
            if drift > config.old_actor_logprob_tolerance:
                raise ValueError('training engine OLD/actor numerical mismatch exceeds qualified tolerance')
            mask = torch.ones_like(current).unsqueeze(0)
            advantage = torch.full_like(mask, row['advantage'])
            weight = torch.tensor([row['prompt_weight']], dtype=torch.float64, device=current.device)
            loss, metrics = policy_loss(old.unsqueeze(0), current.unsqueeze(0), advantage, mask, weight, config)
            if not torch.isfinite(loss):
                raise ValueError('finite upstream PPO loss required')
            detail = dict(task_id=group.task_id, nonce=rollout.nonce, tokens=len(rollout.tokens),
                outcome=rollout.outcome, stop=rollout.stop, reward=float(rollout.outcome == 'correct'),
                advantage=row['advantage'], prompt_weight=row['prompt_weight'],
                loss=float(loss.detach()), old_logprob_mean=float(old.mean()),
                actor_logprob_mean=float(current.detach().mean()), old_actor_max_abs_drift=drift, **metrics)
            losses.append(detail['loss'])
            details.append(detail)
            pending.append(loss)
            if len(pending) == microbatch_rollouts:
                torch.stack(pending).sum().backward()
                accumulator.capture()
                pending.clear()
            record(dict(stage='trajectory-loss', **detail))
            del loss, current, old
        if pending:
            torch.stack(pending).sum().backward()
            accumulator.capture()
            pending.clear()
        norm = accumulator.clip(config.max_gradient_norm)
        guard()
        if _versions(runtime.model) != versions:
            raise ValueError('actor changed before optimizer step')
        step_started = time.monotonic()
        update = optimizer.step(accumulator.buffers)
        if optimizer.global_step != binding.optimizer_step + 1:
            raise ValueError('exactly one optimizer step required')
        return dict(base_report, status='updated', optimizer_step=optimizer.global_step,
            trained_rollouts=len(details), masked_rollouts=len(rows) - len(details),
            sampled_tokens=sum(len(r['rollout'].tokens) for r in rows),
            trained_tokens=sum(r['tokens'] for r in details),
            outcome_counts={outcome: sum(r['rollout'].outcome == outcome for r in rows)
                            for outcome in ('correct', 'wrong', 'unresolved')},
            objective_loss=math.fsum(losses), preclip_gradient_norm=norm,
            maximum_old_actor_logprob_drift=maximum_drift, trajectories=details,
            optimizer=update, preparation_seconds=preparation_seconds,
            optimizer_seconds=time.monotonic() - step_started,
            update_seconds=time.monotonic() - started,
            data_sha256=digest([asdict(group) for group in groups]))
    finally:
        optimizer.zero_grad()
        accumulator.buffers.clear()
