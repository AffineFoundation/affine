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

from .config import TrainerConfig, MASK_UNRESOLVED, TRAIN_UNRESOLVED, PROMPT_MEAN, SC_MIS
from .data import InputBinding, TaskGroup, digest
from .optimizer import FP32Accumulator
from .upstream import reference, policy_loss, aggregate, k3_kl
from .sampler_correction import corrected_terms, selected_candidates


def _versions(model):
    return tuple((name, id(p), p._version) for name, p in model.named_parameters())


def derail_keep(logprobs, *, window=32, threshold=-3., tail=64):
    """Return (kept token count, derail index or None) for one rollout.

    The derail index is the first position i whose trailing window mean
    sampled-token log-prob (positions max(0, i-window+1)..i, requiring a full
    window) is below threshold. Tokens 0..i+tail inclusive are kept.
    """
    values = torch.as_tensor(logprobs, dtype=torch.float64).flatten()
    n = values.numel()
    if n < window:
        return n, None
    means = values.unfold(0, window, 1).mean(-1)
    below = torch.nonzero(means < threshold)
    if below.numel() == 0:
        return n, None
    index = int(below[0, 0]) + window - 1
    return min(n, index + tail + 1), index


def _prompt_weights(rows):
    longest = max(len(row['rollout'].tokens) for row in rows)
    mask = torch.zeros((len(rows), longest), dtype=torch.bool)
    for index, row in enumerate(rows):
        if row['active']:
            mask[index, :row['keep']] = True
    weights = reference().compute_prompt_loss_weights(mask, [row['group'].task_id for row in rows])
    for row, weight in zip(rows, weights, strict=True):
        row['prompt_weight'] = float(weight)


def _layout(groups, config):
    """Derive global prompt weights BEFORE partitioning into microbatches.

    Unresolved (budget-capped or EOS-without-answer) rollouts always enter the
    group mean with reward 0. With TRAIN_UNRESOLVED they are also trained.
    """
    rows = []
    for group in groups:
        rewards = torch.tensor([[config.reward(r.outcome, len(r.tokens))] for r in group.trajectories],
                               dtype=torch.float32)
        advantages, _ = reference().compute_grpo_outcome_advantage(rewards,
            torch.ones_like(rewards), np.array([group.task_id] * config.group_size),
            norm_adv_by_std_in_grpo=False)
        for rollout, advantage in zip(group.trajectories, advantages[:, 0], strict=True):
            active = not (rollout.outcome == 'unresolved' and config.cap_policy == MASK_UNRESOLVED)
            rows.append(dict(group=group, rollout=rollout, advantage=float(advantage), active=active,
                             keep=len(rollout.tokens), derail_index=None,
                             reward=config.reward(rollout.outcome, len(rollout.tokens))))
    _prompt_weights(rows)
    return rows


def train_update(runtime, optimizer, groups, binding: InputBinding, *,
                 config=TrainerConfig(), guard=lambda: None,
                 record=lambda row: None, microbatch_rollouts=1, reference_runtime=None):
    """Advance the caller's leased actor once, after accumulating whole groups.

    `runtime` supplies model, checkpoint_id, vocab_size, eos_token_ids,
    max_context_tokens and score(prompt, output, gradients=bool). The caller
    authenticates the contract and native admission receipts before this API.
    `optimizer` is a persistent CPUAdamW with matching named actor parameters.
    `reference_runtime` (required iff config.kl_coefficient > 0) is a frozen
    runtime whose checkpoint_id equals config.kl_reference_checkpoint.
    With config.sampler_correction == SC_MIS the runtime must also provide
    score_support(prompt, output, top_k=, top_p=) and
    score_candidates(prompt, output, indices, gradients=).
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
    if config.kl_coefficient > 0:
        if reference_runtime is None or reference_runtime.checkpoint_id != config.kl_reference_checkpoint:
            raise ValueError('KL anchoring requires the configured frozen reference runtime')
        if reference_runtime is runtime or getattr(reference_runtime, 'model', None) is runtime.model:
            raise ValueError('KL reference must be a separate frozen model')
    elif reference_runtime is not None:
        raise ValueError('reference runtime supplied without a KL objective')
    correcting = config.sampler_correction == SC_MIS
    if correcting and not (callable(getattr(runtime, 'score_support', None)) and
                           callable(getattr(runtime, 'score_candidates', None))):
        raise ValueError('sampler correction requires support replay and candidate scoring')
    expected = tuple((name, id(p)) for name, p in runtime.model.named_parameters() if p.requires_grad)
    if expected != tuple((name, id(p)) for name, p in optimizer.parameters):
        raise ValueError('optimizer must own exact trainable actor parameters')
    for group in groups:
        if not isinstance(group, TaskGroup):
            raise ValueError('native trusted TaskGroup required')
        group.validate(config.group_size, vocab_size=runtime.vocab_size, eos_token_ids=set(runtime.eos_token_ids),
                       unresolved_failures=config.cap_policy == TRAIN_UNRESOLVED)
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
        row['old'] = row['ref'] = row['support'] = None
        if not row['active']:
            continue
        prompt, tokens = row['group'].prompt, row['rollout'].tokens
        with torch.no_grad():
            if correcting:
                old, indices, q = runtime.score_support(prompt, tokens, top_k=config.top_k, top_p=config.top_p)
                indices, q = indices.detach().cpu(), q.detach().cpu()
                selected = selected_candidates(indices, q, tokens)
                if indices.shape != q.shape or indices.shape[0] != len(tokens):
                    raise ValueError('replayed sampler support for every position')
                row['support'] = (indices, q, selected)
            else:
                old = runtime.score(prompt, tokens, gradients=False)
        if old.shape != (len(tokens),) or not torch.isfinite(old).all():
            raise ValueError('trusted OLD requires finite selected-token probabilities')
        row['old'] = old.detach().to(device='cpu', dtype=torch.float32).clone()
        del old
        if (config.derail_mask and row['rollout'].stop == 'max_tokens' and row['advantage'] < 0):
            row['keep'], row['derail_index'] = derail_keep(row['old'], window=config.derail_window,
                threshold=config.derail_threshold, tail=config.derail_tail)
        if row['keep'] < len(tokens):
            # Causal LM: positions before keep do not depend on later tokens, so
            # the gradient pass (and KL reference) run on the kept prefix only.
            # Re-score OLD on that exact prefix so OLD/actor share kernel shapes.
            tokens = tokens[:row['keep']]
            with torch.no_grad():
                if correcting:
                    old, indices, q = runtime.score_support(prompt, tokens, top_k=config.top_k, top_p=config.top_p)
                    indices, q = indices.detach().cpu(), q.detach().cpu()
                    row['support'] = (indices, q, selected_candidates(indices, q, tokens))
                else:
                    old = runtime.score(prompt, tokens, gradients=False)
            if old.shape != (len(tokens),) or not torch.isfinite(old).all():
                raise ValueError('trusted OLD requires finite selected-token probabilities')
            row['old'] = old.detach().to(device='cpu', dtype=torch.float32).clone()
            del old
        row['tokens'] = tokens
        if config.kl_coefficient > 0:
            with torch.no_grad():
                ref = reference_runtime.score(prompt, tokens, gradients=False)
            if ref.shape != (len(tokens),) or not torch.isfinite(ref).all():
                raise ValueError('finite reference selected-token probabilities required')
            row['ref'] = ref.detach().to(device='cpu', dtype=torch.float32).clone()
            del ref
    if config.loss_agg_mode == PROMPT_MEAN:
        _prompt_weights(rows)
    active_rollouts = sum(1 for row in rows if row['active'] and row['keep'] > 0)
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
                    tokens=len(rollout.tokens), reward=row['reward'], advantage=row['advantage']))
                continue
            if _versions(runtime.model) != versions:
                raise ValueError('actor mutated before the single optimizer boundary')
            with torch.enable_grad():
                if correcting:
                    current, candidates = runtime.score_candidates(group.prompt, row['tokens'],
                                                                   row['support'][0], gradients=True)
                else:
                    current = runtime.score(group.prompt, row['tokens'], gradients=True)
            if current.shape != row['old'].shape or not torch.isfinite(current).all():
                raise ValueError('finite differentiable current token probabilities required')
            old = row['old'].to(device=current.device, dtype=current.dtype)
            drift = float((current.detach() - old).abs().max())
            maximum_drift = max(maximum_drift, drift)
            if drift > config.old_actor_logprob_tolerance:
                raise ValueError('training engine OLD/actor numerical mismatch exceeds qualified tolerance')
            mask = torch.zeros_like(current).unsqueeze(0)
            mask[0, :row['keep']] = 1
            advantage = torch.full_like(mask, row['advantage'])
            weight = torch.tensor([row['prompt_weight']], dtype=torch.float64, device=current.device)
            info = dict(global_batch_size=active_rollouts)
            if correcting:
                indices, q, selected = row['support']
                inside = (selected >= 0).to(device=current.device, dtype=mask.dtype).unsqueeze(0)
                mask = mask * inside
                term, extra = corrected_terms(candidates, q.to(candidates.device), selected.to(candidates.device),
                    lower=config.sampler_ratio_lower, upper=config.sampler_ratio_upper)
                loss = aggregate((-advantage[0] * term).unsqueeze(0), mask, weight, config, **info)
                metrics = {'sampler/outside_support': int((selected < 0).sum()),
                           'sampler/selected_weight_mean': float(extra['selected_weight'].mean())}
                del candidates
            else:
                loss, metrics = policy_loss(old.unsqueeze(0), current.unsqueeze(0), advantage, mask, weight,
                                            config, **info)
            kl_mean = None
            if config.kl_coefficient > 0:
                kld = k3_kl(current.unsqueeze(0), row['ref'].to(device=current.device, dtype=current.dtype).unsqueeze(0))
                loss = loss + config.kl_coefficient * aggregate(kld, mask, weight, config, **info)
                kl_mean = float((kld.detach() * mask).sum() / mask.sum().clamp_min(1))
            if not torch.isfinite(loss):
                raise ValueError('finite upstream PPO loss required')
            detail = dict(task_id=group.task_id, nonce=rollout.nonce, tokens=len(rollout.tokens),
                outcome=rollout.outcome, stop=rollout.stop, reward=row['reward'],
                advantage=row['advantage'], prompt_weight=row['prompt_weight'],
                loss=float(loss.detach()), old_logprob_mean=float(old.mean()),
                actor_logprob_mean=float(current.detach().mean()), old_actor_max_abs_drift=drift,
                loss_tokens=int(mask.sum()), derail_index=row['derail_index'], k3_kl_mean=kl_mean, **metrics)
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
            loss_tokens=sum(r['loss_tokens'] for r in details),
            derailed_rollouts=sum(r['derail_index'] is not None for r in details),
            unresolved_trained_rollouts=sum(r['outcome'] == 'unresolved' for r in details),
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
