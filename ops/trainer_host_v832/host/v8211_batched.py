"""v8.2.11: v8.2.5 batched GRPO update + operator-selected loss aggregation + per-epoch diagnostics.

v8.2.11 delta (Jacob t586u 2026-10-24), set by the ROOT-admitted v8.2.11 host from engine.loss_agg_mode /
engine.loss_scale:
  * LOSS_AGG_MODE == 'config' (default): the signed trainer_config's aggregation (live: seq-mean-token-sum-norm,
    loss_scale_factor 8192), byte-for-byte the v8.2.5 loss. LOSS_SCALE must be 1.
  * LOSS_AGG_MODE == 'seq-mean-token-mean' (verl, original GRPO): each rollout's clipped PG term and k3 KL term
    are averaged over that rollout's own kept response tokens, then averaged over the active rollouts of the
    batch (same pinned upstream agg_loss, mode 'seq-mean-token-mean', global_batch_size = active rollouts),
    times the constant LOSS_SCALE. Advantages, rewards (incl. overlong soft penalty), KL coef/reference, clip,
    derail mask, lr, gating and nonces are untouched; the signed trainer_config document is reported unchanged.
  * report['epoch_metrics']: capped (stop=max_tokens) share of gradient mass (|adv| x token weight under the
    active aggregation), token-mean logprob on passes / capped, pre-clip grad norm, cap rate overall and on
    MATH levels 1-3, PG clip fraction (ratio==1 by construction: actor-as-OLD), and whether grad clipping fired.
  * _reference_pass/_gradient_pass are the exact update stages, reused by the offline replay (no optimizer).

v8.2.5: v8.2.4 micro-batched single-step GRPO update + config-gated unresolved-rollout drop.

v8.2.5 delta (only when DROP_UNRESOLVED is True; set by the ROOT-admitted host launcher from
engine.drop_unresolved): rollouts with outcome == 'unresolved' (budget-capped or EOS without a
gradeable final answer) are excluded from the loss (inactive rows: no gradient, not counted in
global_batch_size / prompt weights). GRPO advantages are recomputed with the same pinned upstream
compute_grpo_outcome_advantage over the RESOLVED rollouts of each group only (mean baseline over
resolved rewards; no std normalisation, as before). A group with fewer than 2 resolved rollouts is
dropped from the loss entirely (all rows inactive). group_count / rollout_count / admitted groups
are unchanged (all admitted groups are still reported, so host validate_result holds); dropped rows
are counted in masked_rollouts and in report['unresolved_filter']. With DROP_UNRESOLVED False the
update is byte-for-byte the v8.2.4 path.

v8.2.4 notes (unchanged):

Differences from training.train_update (v8.2.3), all objective-preserving:
  * OLD is the actor itself: with ppo_epochs=1 and exactly one optimizer step at
    the end, the trusted OLD prefill equals the actor forward (production drift
    0.0). OLD := current.detach(), so ratio == 1 exactly and the dual-clip PPO
    gradient is -A * grad(log p), as today.
  * The derail mask is computed from current.detach() inside the gradient pass
    and applied as a loss mask over the kept prefix (causal LM: identical to
    rescoring the prefix).
  * KL-reference (no grad) and actor (grad) passes run several rollouts per
    forward: 'padded' = right-padded length-bucketed batch (works on the pinned
    torch/eager engine), 'packed' = one varlen sequence with cu_seqlens
    (requires fla + causal-conv1d kernels and sdpa/flash full attention).
  * Per-rollout losses are produced by the SAME pinned upstream policy_loss /
    aggregate / k3_kl calls with the same global normalizers; the micro-batch
    loss is their sum, so d(loss)/d(theta) equals the v1 per-rollout sum.
"""
import math
import time
import torch
from torch.utils.checkpoint import checkpoint

from subnet.mimo_grpo.config import MASK_UNRESOLVED, TRAIN_UNRESOLVED, PROMPT_MEAN, SC_MIS
from subnet.mimo_grpo.data import TaskGroup, digest
from subnet.mimo_grpo.optimizer import FP32Accumulator
from subnet.mimo_grpo.upstream import policy_loss, aggregate, k3_kl
from subnet.mimo_grpo.training import _layout, _prompt_weights, derail_keep, _versions
from subnet.mimo_grpo.upstream import reference
import numpy as np

DROP_UNRESOLVED = False   # v8.2.5: set only by mimo_host_trainer_v825.install from engine['drop_unresolved']
LOSS_AGG_MODE = 'config'   # v8.2.11: set only by mimo_host_trainer_v8211.install from engine['loss_agg_mode']
LOSS_SCALE = 1.0           # v8.2.11: engine['loss_scale'] (constant multiplier, seq-mean-token-mean only)
LOSS_AGG_MODES = ('config', 'seq-mean-token-mean')
SEQ_MEAN = 'seq-mean-token-mean'
TASK_INDEX = {}            # v8.2.11: task_id -> (env_id, index), recorded by the host from the prepared dispatch
MATH_LEVELS = {}           # v8.2.11: str(index) -> MATH level 1..5 (host file math_levels.json)


class _AggView:
    """Trainer config with only the loss aggregation replaced (read-only view for policy_loss/aggregate)."""
    def __init__(self, config, mode):
        object.__setattr__(self, '_config', config)
        object.__setattr__(self, 'loss_agg_mode', mode)
        object.__setattr__(self, 'loss_scale_factor', 0)

    def __getattr__(self, name):
        return getattr(self._config, name)

    def __setattr__(self, name, value):
        raise AttributeError('read-only aggregation view')


def loss_aggregation(config):
    """(config used by the pinned loss calls, constant loss multiplier, report descriptor)."""
    if LOSS_AGG_MODE not in LOSS_AGG_MODES:
        raise ValueError('unsupported engine loss aggregation')
    scale = float(LOSS_SCALE)
    if not math.isfinite(scale) or not 0 < scale <= 1024:
        raise ValueError('bounded positive finite loss scale')
    if LOSS_AGG_MODE == 'config':
        if scale != 1.0:
            raise ValueError('config aggregation has no extra loss scale')
        return config, 1.0, dict(mode='config', loss_agg_mode=config.loss_agg_mode,
                                 loss_scale_factor=config.loss_scale_factor, loss_scale=1.0)
    return _AggView(config, SEQ_MEAN), scale, dict(mode=SEQ_MEAN, loss_agg_mode=SEQ_MEAN,
        signed_config_loss_agg_mode=config.loss_agg_mode, loss_scale=scale)


def token_weight(row, config, desc, active_rollouts):
    """Total loss weight of one rollout's kept tokens, as applied by the active aggregation (x |adv| = mass)."""
    keep = row['keep'] if row['active'] else 0
    if not keep:
        return 0.0
    if desc['mode'] == SEQ_MEAN:
        return desc['loss_scale'] / active_rollouts
    if config.loss_agg_mode == 'prompt-mean':
        return row['prompt_weight'] * keep
    return keep / (active_rollouts * config.loss_scale_factor)


def epoch_metrics(rows, details, config, desc, active_rollouts, norm):
    def share(selector):
        mass = [abs(r['advantage']) * token_weight(r, config, desc, active_rollouts) for r in rows]
        total = math.fsum(mass)
        return math.fsum(m for m, r in zip(mass, rows) if selector(r)) / total if total else 0.0

    def logprob(selector):
        picked = [d for d in details.values() if selector(d)]
        tokens = sum(d['loss_tokens'] for d in picked)
        value = math.fsum(d['old_logprob_mean'] * d['loss_tokens'] for d in picked) / tokens if tokens else None
        return dict(token_mean=value, rollouts=len(picked), tokens=tokens)

    capped = lambda r: r['rollout'].stop == 'max_tokens'
    n = len(rows)
    by_level = {}
    for r in rows:
        env_index = TASK_INDEX.get(r['group'].task_id)
        level = MATH_LEVELS.get(str(env_index[1])) if env_index else None
        key = 'L%d' % level if level else 'unknown'
        c = by_level.setdefault(key, [0, 0])
        c[0] += 1; c[1] += int(capped(r))
    low = [sum(by_level.get('L%d' % k, [0, 0])[i] for k in (1, 2, 3)) for i in (0, 1)]
    tokens = sum(d['loss_tokens'] for d in details.values())
    clipfrac = (math.fsum(d.get('actor/pg_clipfrac', 0.) * d['loss_tokens'] for d in details.values()) / tokens
                if tokens else 0.0)
    return dict(version='affine-mimo-epoch-metrics-v8211', aggregation=desc,
        capped_gradient_share=share(capped),
        negative_advantage_gradient_share=share(lambda r: r['advantage'] < 0),
        unresolved_gradient_share=share(lambda r: r['rollout'].outcome == 'unresolved'),
        logprob_pass=logprob(lambda d: d['outcome'] == 'correct'),
        logprob_capped=logprob(lambda d: d['stop'] == 'max_tokens'),
        logprob_all=logprob(lambda d: True),
        preclip_gradient_norm=norm, max_gradient_norm=config.max_gradient_norm,
        gradient_clip_applied=bool(norm is not None and norm > config.max_gradient_norm),
        cap_rate=sum(capped(r) for r in rows) / n if n else 0.0, capped_rollouts=sum(capped(r) for r in rows),
        rollouts=n, cap_rate_math_l1_3=low[1] / low[0] if low[0] else None, rollouts_math_l1_3=low[0],
        cap_rate_by_level={k: dict(rollouts=v[0], capped=v[1], rate=v[1] / v[0]) for k, v in sorted(by_level.items())},
        pg_clipfrac_token_mean=clipfrac,
        pg_clipfrac_note='actor-as-OLD with one PPO epoch: ratio == 1 exactly, so PPO clipping never binds')
DROP_POLICY = 'affine-drop-unresolved-resolved-only-advantage-min2-v1'

ENGINES = ('padded', 'packed')


def layout_drop_unresolved(groups, config):
    """v8.2.5: rows like training._layout, but unresolved rollouts never enter the loss and the
    group baseline is computed over resolved rollouts only; groups with < 2 resolved are dropped."""
    rows = []
    stats = dict(enabled=True, policy=DROP_POLICY, groups_total=len(groups), dropped_groups=0,
                 dropped_unresolved_rollouts=0, dropped_resolved_rollouts_in_dropped_groups=0,
                 dropped_unresolved_by_stop={}, dropped_unresolved_tokens=0, kept_rollouts=0)
    for group in groups:
        resolved = [r for r in group.trajectories if r.outcome != 'unresolved']
        keep_group = len(resolved) >= 2
        advantage = {}
        if keep_group:
            rewards = torch.tensor([[config.reward(r.outcome, len(r.tokens))] for r in resolved], dtype=torch.float32)
            values, _ = reference().compute_grpo_outcome_advantage(rewards,
                torch.ones_like(rewards), np.array([group.task_id] * len(resolved)),
                norm_adv_by_std_in_grpo=False)
            for rollout, value in zip(resolved, values[:, 0], strict=True):
                advantage[id(rollout)] = float(value)
        else:
            stats['dropped_groups'] += 1
            stats['dropped_resolved_rollouts_in_dropped_groups'] += len(resolved)
        for rollout in group.trajectories:
            unresolved = rollout.outcome == 'unresolved'
            if unresolved:
                stats['dropped_unresolved_rollouts'] += 1
                stats['dropped_unresolved_tokens'] += len(rollout.tokens)
                stop = str(rollout.stop)
                stats['dropped_unresolved_by_stop'][stop] = stats['dropped_unresolved_by_stop'].get(stop, 0) + 1
            active = keep_group and not unresolved
            stats['kept_rollouts'] += int(active)
            rows.append(dict(group=group, rollout=rollout, advantage=advantage.get(id(rollout), 0.0), active=active,
                             keep=len(rollout.tokens), derail_index=None,
                             reward=config.reward(rollout.outcome, len(rollout.tokens))))
    if not stats['kept_rollouts']:
        raise ValueError('drop_unresolved left no trainable rollouts in the batch')
    _prompt_weights(rows)
    return rows, stats


def plan_microbatches(lengths, token_budget, engine, max_rows=64):
    """Greedy length-sorted bucketing. padded cost = rows*max_len, packed cost = sum."""
    if engine not in ENGINES or type(token_budget) is not int or token_budget < 1:
        raise ValueError('micro-batch engine/token budget')
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    batches, cur, cur_max, cur_sum = [], [], 0, 0
    for i in order:
        n = lengths[i]
        if n > token_budget:
            raise ValueError('single rollout exceeds micro-batch token budget')
        new_max, new_sum = max(cur_max, n), cur_sum + n
        cost = new_max * (len(cur) + 1) if engine == 'padded' else new_sum
        if cur and (cost > token_budget or len(cur) >= max_rows):
            batches.append(cur)
            cur, new_max, new_sum = [], n, n
        cur.append(i)
        cur_max, cur_sum = new_max, new_sum
    if cur:
        batches.append(cur)
    return batches


def batched_log_probs(runtime, items, *, gradients, engine, logit_chunk=2048):
    """Selected raw log-probs for each (prompt, output) pair, one forward per call."""
    runtime.guard()
    if gradients and not runtime.trainable:
        raise ValueError('inference runtime cannot train')
    model, device = runtime.model, runtime.device
    seqs = []
    for prompt, output in items:
        prompt, output = runtime._tokens(prompt), runtime._tokens(output)
        if len(prompt) + len(output) > runtime.max_context_tokens:
            raise ValueError('response context limit')
        seqs.append(prompt + output)
    with torch.set_grad_enabled(gradients):
        if engine == 'padded':
            width = max(len(s) for s in seqs)
            ids = torch.zeros((len(seqs), width), dtype=torch.long)
            mask = torch.zeros((len(seqs), width), dtype=torch.long)
            for row, s in enumerate(seqs):
                ids[row, :len(s)] = torch.tensor(s)
                mask[row, :len(s)] = 1
            ids, mask = ids.to(device), mask.to(device)
            if len(seqs) == 1:
                hidden = model.model(ids, use_cache=False).last_hidden_state
            else:
                hidden = model.model(ids, attention_mask=mask, use_cache=False).last_hidden_state
            pieces = [hidden[row, len(p) - 1:len(p) + len(o) - 1] for row, (p, o) in enumerate(items)]
        elif engine == 'packed':
            lens = [len(s) for s in seqs]
            ids = torch.tensor([t for s in seqs for t in s], device=device)[None]
            pos = torch.cat([torch.arange(n) for n in lens])[None].to(device)
            cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
            cu[1:] = torch.tensor(lens).cumsum(0)
            seq_idx = torch.cat([torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(lens)])[None].to(device)
            offsets = cu.tolist()
            cu = cu.to(device)
            hidden = model.model(ids, position_ids=pos, use_cache=False, cu_seq_lens_q=cu, cu_seq_lens_k=cu,
                                 max_length_q=max(lens), max_length_k=max(lens), seq_idx=seq_idx).last_hidden_state[0]
            pieces = [hidden[offsets[r] + len(p) - 1:offsets[r] + len(p) + len(o) - 1] for r, (p, o) in enumerate(items)]
        else:
            raise ValueError('engine')
        states = torch.cat(pieces)
        targets = torch.tensor([t for _, o in items for t in o], device=device)

        def selected(h, t):
            logits = model.lm_head(h).float()
            return logits.gather(-1, t[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        chunks = []
        for start in range(0, len(targets), logit_chunk):
            h, t = states[start:start + logit_chunk], targets[start:start + logit_chunk]
            chunks.append(checkpoint(selected, h, t, use_reentrant=False) if gradients else selected(h, t))
        flat = torch.cat(chunks)
    return list(torch.split(flat, [len(o) for _, o in items]))


def _reference_pass(rows, plan, config, reference_runtime, *, engine, logit_chunk, guard=lambda: None):
    """KL-reference selected-token log-probs (no grad), stored per row on CPU (v8.2.5 stage, unchanged)."""
    for row in rows:
        row['ref'] = None
    if config.kl_coefficient > 0:
        for batch in plan:
            guard()
            with torch.no_grad():
                refs = batched_log_probs(reference_runtime, [(rows[i]['group'].prompt, rows[i]['rollout'].tokens)
                                                             for i in batch], gradients=False, engine=engine,
                                         logit_chunk=logit_chunk)
            for i, ref in zip(batch, refs):
                if ref.shape != (len(rows[i]['rollout'].tokens),) or not torch.isfinite(ref).all():
                    raise ValueError('finite reference selected-token probabilities required')
                rows[i]['ref'] = ref.detach().to(device='cpu', dtype=torch.float32).clone()
            del refs


def _gradient_pass(runtime, rows, plan, config, agg_config, loss_scale, active_rollouts, accumulator, versions,
                   details, losses, *, engine, logit_chunk, guard=lambda: None, record=lambda row: None):
    """Accumulate d(loss)/d(theta) over every micro-batch (v8.2.5 stage; aggregation from loss_aggregation)."""
    for i, row in enumerate(rows):
        if not row['active']:
            record(dict(stage='masked-unresolved', task_id=row['group'].task_id, nonce=row['rollout'].nonce,
                        tokens=len(row['rollout'].tokens), reward=row['reward'], advantage=row['advantage']))
    for batch in plan:
        guard()
        if _versions(runtime.model) != versions:
            raise ValueError('actor mutated before the single optimizer boundary')
        with torch.enable_grad():
            currents = batched_log_probs(runtime, [(rows[i]['group'].prompt, rows[i]['rollout'].tokens)
                                                   for i in batch], gradients=True, engine=engine,
                                         logit_chunk=logit_chunk)
        pending = []
        for i, current in zip(batch, currents):
            row = rows[i]
            rollout, group = row['rollout'], row['group']
            if current.shape != (len(rollout.tokens),) or not torch.isfinite(current).all():
                raise ValueError('finite differentiable current token probabilities required')
            old = current.detach()
            if config.derail_mask and rollout.stop == 'max_tokens' and row['advantage'] < 0:
                row['keep'], row['derail_index'] = derail_keep(old.to('cpu', torch.float32), window=config.derail_window,
                    threshold=config.derail_threshold, tail=config.derail_tail)
            mask = torch.zeros_like(current).unsqueeze(0)
            mask[0, :row['keep']] = 1
            advantage = torch.full_like(mask, row['advantage'])
            weight = torch.tensor([row['prompt_weight']], dtype=torch.float64, device=current.device)
            info = dict(global_batch_size=active_rollouts)
            loss, metrics = policy_loss(old.unsqueeze(0), current.unsqueeze(0), advantage, mask, weight,
                                        agg_config, **info)
            kl_mean = None
            if config.kl_coefficient > 0:
                kld = k3_kl(current.unsqueeze(0), row['ref'].to(device=current.device, dtype=current.dtype).unsqueeze(0))
                loss = loss + config.kl_coefficient * aggregate(kld, mask, weight, agg_config, **info)
                kl_mean = float((kld.detach() * mask).sum() / mask.sum().clamp_min(1))
            if loss_scale != 1.0:
                loss = loss * loss_scale
            if not torch.isfinite(loss):
                raise ValueError('finite upstream PPO loss required')
            detail = dict(task_id=group.task_id, nonce=rollout.nonce, tokens=len(rollout.tokens),
                outcome=rollout.outcome, stop=rollout.stop, reward=row['reward'],
                advantage=row['advantage'], prompt_weight=row['prompt_weight'],
                loss=float(loss.detach()), old_logprob_mean=float(old[:row['keep']].mean()) if row['keep'] else 0.,
                actor_logprob_mean=float(old[:row['keep']].mean()) if row['keep'] else 0.,
                old_actor_max_abs_drift=0.0,
                loss_tokens=int(mask.sum()), derail_index=row['derail_index'], k3_kl_mean=kl_mean, **metrics)
            details[i] = detail
            losses[i] = detail['loss']
            pending.append(loss)
            record(dict(stage='trajectory-loss', **detail))
        torch.stack(pending).sum().backward()
        accumulator.capture()
        del pending, currents


def train_update_batched(runtime, optimizer, groups, binding, *, config, guard=lambda: None,
                         record=lambda row: None, reference_runtime=None, engine='padded',
                         token_budget=16384, max_rows=64, logit_chunk=2048, timings=None):
    config.validate()
    binding.validate()
    agg_config, loss_scale, aggregation = loss_aggregation(config)   # v8.2.11: fail before any work
    groups = tuple(groups)
    if config.ppo_epochs != 1:
        raise ValueError('actor-as-OLD requires exactly one PPO epoch / one optimizer step')
    if config.sampler_correction == SC_MIS:
        raise ValueError('sampler correction not supported by the batched engine')
    if config.loss_agg_mode == PROMPT_MEAN and config.derail_mask:
        raise ValueError('prompt-mean weights need derail keep before the gradient pass')
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
    base_report = dict(version='affine-mimo-grpo-update-v1', config=config.document(), input=dict(
        run=binding.run, epoch=binding.epoch, checkpoint=binding.checkpoint,
        contract_sha256=binding.contract_sha256, optimizer_step=binding.optimizer_step),
        admitted_groups=[dict(task_id=g.task_id, admission_sha256=g.admission_sha256) for g in groups],
        group_count=len(groups), rollout_count=len(groups) * config.group_size,
        input_checkpoint=binding.checkpoint, optimizer_step_before=optimizer.global_step)
    if not groups:
        return dict(base_report, status='no-update', reason='no-eligible-complete-groups',
                    optimizer_step=optimizer.global_step)
    timings = {} if timings is None else timings
    started = time.monotonic()
    runtime.model.eval()
    versions = _versions(runtime.model)
    if DROP_UNRESOLVED:
        rows, unresolved_filter = layout_drop_unresolved(groups, config)
    else:
        rows, unresolved_filter = _layout(groups, config), dict(enabled=False)
    record(dict(stage='unresolved-filter', **unresolved_filter))
    active = [i for i, row in enumerate(rows) if row['active']]
    lengths = [len(rows[i]['group'].prompt) + len(rows[i]['rollout'].tokens) for i in active]
    plan = [[active[j] for j in batch] for batch in plan_microbatches(lengths, token_budget, engine, max_rows)]
    preparation_started = time.monotonic()
    _reference_pass(rows, plan, config, reference_runtime, engine=engine, logit_chunk=logit_chunk, guard=guard)
    active_rollouts = len(active)
    preparation_seconds = time.monotonic() - preparation_started
    if _versions(runtime.model) != versions:
        raise ValueError('input actor changed during reference preparation')
    optimizer.zero_grad()
    accumulator = FP32Accumulator(runtime.model.named_parameters())
    details, losses = {}, {}
    gradient_started = time.monotonic()
    try:
        _gradient_pass(runtime, rows, plan, config, agg_config, loss_scale, active_rollouts, accumulator, versions,
                       details, losses, engine=engine, logit_chunk=logit_chunk, guard=guard, record=record)
        timings['gradient_seconds'] = time.monotonic() - gradient_started
        norm = accumulator.clip(config.max_gradient_norm)
        guard()
        if _versions(runtime.model) != versions:
            raise ValueError('actor changed before optimizer step')
        step_started = time.monotonic()
        update = optimizer.step(accumulator.buffers)
        if optimizer.global_step != binding.optimizer_step + 1:
            raise ValueError('exactly one optimizer step required')
        ordered = [details[i] for i in sorted(details)]
        return dict(base_report, status='updated', optimizer_step=optimizer.global_step,
            trained_rollouts=len(ordered), masked_rollouts=len(rows) - len(ordered),
            sampled_tokens=sum(len(r['rollout'].tokens) for r in rows),
            trained_tokens=sum(r['tokens'] for r in ordered),
            loss_tokens=sum(r['loss_tokens'] for r in ordered),
            derailed_rollouts=sum(r['derail_index'] is not None for r in ordered),
            unresolved_trained_rollouts=sum(r['outcome'] == 'unresolved' for r in ordered),
            outcome_counts={outcome: sum(r['rollout'].outcome == outcome for r in rows)
                            for outcome in ('correct', 'wrong', 'unresolved')},
            objective_loss=math.fsum(losses[i] for i in sorted(losses)), preclip_gradient_norm=norm,
            maximum_old_actor_logprob_drift=0.0, trajectories=ordered,
            optimizer=update, preparation_seconds=preparation_seconds,
            optimizer_seconds=time.monotonic() - step_started,
            update_seconds=time.monotonic() - started,
            microbatch=dict(engine=engine, token_budget=token_budget, max_rows=max_rows, count=len(plan)),
            unresolved_filter=unresolved_filter,
            data_sha256=digest([__import__('dataclasses').asdict(group) for group in groups]),
            loss_aggregation=aggregation,
            epoch_metrics=epoch_metrics(rows, details, config, aggregation, active_rollouts, norm))
    finally:
        optimizer.zero_grad()
        accumulator.buffers.clear()
