"""One accumulated full-model update, with memory bounded to one rollout graph."""
from dataclasses import asdict
import math
import time
import torch

from .core import require, group_eligibility
from .objectives import group_advantages, grpo_trajectory_loss, preference_coefficients
from .scoring import score
from research.group_ratio_guard.guard import Thresholds, Trajectory, assess


def train_update(model, reference_model, optimizer, groups, *, objective, group_size,
                 guard, record, numerical_guard=Thresholds()):
    from subnet.fp32_gradient_accumulation import FP32GradientAccumulator
    from subnet.optimizer_threading import step as threaded_step, POLICY
    objective.validate()
    require(bool(groups), 'nonempty admitted mixed groups')
    stamp = groups[0].stamp
    require(all(g.stamp == stamp and group_eligibility(g, group_size)['eligible'] for g in groups),
            'one unchanged behavior checkpoint and only mixed complete groups')
    require(len({g.task_id for g in groups}) == len(groups), 'no duplicate task in update')
    require(optimizer.global_step == stamp.optimizer_step, 'optimizer matches collection')
    require(model is not reference_model and not any(p.requires_grad for p in reference_model.parameters()),
            'independent frozen reference')
    # Prefill same numerical training mode once before any gradient; old sampler
    # probabilities remain the true denominator, not this convenient prepass.
    numerical_guard.validate()
    require(numerical_guard.clip_epsilon == objective.clip_epsilon, 'matching objective/diagnostic clipping')
    scalar_rows = []
    all_drifts = []
    initial_rows = []
    for group in groups:
        advantages, _ = group_advantages(group, group_size=group_size, epsilon=objective.advantage_epsilon)
        rows = []
        for rollout in group.rollouts:
            guard()
            if rollout.outcome == 'unresolved':
                rows.append(None)
                continue
            # Gradient-enabled pass matches the subsequent checkpointed kernel.
            values = score(model, group.prompt, rollout.output, stamp.policy, gradients=True)
            current = values.detach().cpu().tolist()
            del values
            drifts = [abs(a-b) for a, b in zip(current, rollout.behavior_logprobs)]
            drift = max(drifts)
            all_drifts.extend(drifts)
            record(dict(stage='generation_scoring_drift',task_id=group.task_id,nonce=rollout.attempt,
                        tokens=len(drifts),mean_abs_logratio=sum(drifts)/len(drifts),
                        p99_abs_logratio=float(torch.quantile(torch.tensor(drifts,dtype=torch.float64),.99)),
                        max_abs_logratio=drift))
            initial_rows.append(Trajectory(group.task_id, rollout.attempt, rollout.outcome,
                advantages[rollout.attempt], tuple(rollout.behavior_logprobs), tuple(current)))
            frozen = (score(reference_model, group.prompt, rollout.output, stamp.policy, gradients=False)
                      .detach().cpu().tolist()) if objective.mode == 'grpo' and objective.kl_beta else current
            rows.append(dict(current=current, frozen=frozen, drift=drift))
        scalar_rows.append(rows)
    optimizer.zero_grad()
    # Inspect the whole intended update before any gradient or optimizer change.
    # This is a training-usability bound, never an inference-proof verdict.
    assessment = assess(initial_rows, numerical_guard)
    record(dict(stage='initial_ratio_assessment', **assessment))
    require(assessment['passed'], 'initial ratio usability guard rejected update')
    drift_summary = dict(assessment['aggregate'], thresholds=assessment['thresholds'])
    record(dict(stage='all_initial_ratios_admitted', **drift_summary))
    accumulator = FP32GradientAccumulator(model.named_parameters())
    losses, details, preference_losses = [], [], []
    started = time.monotonic()
    try:
        for group, prepass in zip(groups, scalar_rows):
            advantages, stats = group_advantages(group, group_size=group_size,
                                                epsilon=objective.advantage_epsilon)
            coefficients = None
            if objective.mode == 'group_preference':
                means = [sum(r['current'])/len(r['current']) if r else 0. for r in prepass]
                # Epoch/update input checkpoint reference uses SAME scoring
                # kernel, while candidate ratios use actual generation LP.
                coefficients, pair_details = preference_coefficients(group, means, means,
                    group_size=group_size, beta=objective.preference_beta, task_count=len(groups))
                preference_losses.append(pair_details['loss'])
                record(dict(stage='preference_group', task_id=group.task_id, **pair_details))
            for rollout, row, advantage in zip(group.rollouts, prepass, advantages):
                guard()
                if row is None:
                    record(dict(stage='masked_unresolved', task_id=group.task_id, nonce=rollout.attempt,
                                stop=rollout.stop, outcome=rollout.outcome, tokens=len(rollout.output),
                                reward=0., advantage=advantage, group_statistics=stats))
                    continue
                current = score(model, group.prompt, rollout.output, stamp.policy, gradients=True)
                require(torch.equal(current.detach(), torch.tensor(row['current'], device=current.device, dtype=current.dtype)),
                        'unchanged model prepass/gradient numerical parity')
                if objective.mode == 'grpo':
                    loss, detail = grpo_trajectory_loss(current, rollout.behavior_logprobs, row['frozen'],
                        advantage=advantage, objective=objective, task_count=len(groups), group_size=group_size)
                else:
                    loss = current.mean() * coefficients[rollout.attempt]
                    detail = dict(loss=float(loss.detach()), derivative_coefficient=coefficients[rollout.attempt],
                                  surrogate_for_exact_first_derivative=True)
                require(bool(torch.isfinite(loss)), 'finite objective')
                losses.append(float(loss.detach()))
                loss.backward()
                accumulator.capture()
                detail.update(task_id=group.task_id, nonce=rollout.attempt, outcome=rollout.outcome,
                              stop=rollout.stop, tokens=len(rollout.output), behavior_logprobs=rollout.behavior_logprobs,
                              current_logprobs=row['current'], reference_logprobs=row['frozen'],
                              generation_scoring_max_logratio=row['drift'])
                details.append(detail)
                record(dict(stage='trajectory_backward', **detail))
                del loss, current
        require(accumulator.microsteps == len(details) > 0, 'all and only resolved trajectories accumulated')
        preclip = accumulator.clip(1.)
        guard()
        optimizer_started = time.monotonic()
        precision, threading = threaded_step(torch, optimizer, accumulator.gradients(), POLICY)
        require(optimizer.global_step == stamp.optimizer_step + 1, 'one optimizer advancement')
        objective_loss = math.fsum(preference_losses if objective.mode == 'group_preference' else losses)
        return dict(objective=asdict(objective), group_count=len(groups),
            rollout_count=len(groups)*group_size, trained_rollouts=len(details),
            masked_rollouts=len(groups)*group_size-len(details),
            total_loss=objective_loss, objective_loss=objective_loss,
            gradient_surrogate_loss=math.fsum(losses), preclip_gradient_norm=preclip,
            outcome_counts={outcome:sum(r.outcome==outcome for g in groups for r in g.rollouts)
                            for outcome in ('correct','wrong','unresolved')},
            sampled_tokens=sum(len(r.output) for g in groups for r in g.rollouts),
            trained_tokens=sum(len(r.output) for g in groups for r in g.rollouts if r.outcome!='unresolved'),
            optimizer_seconds=time.monotonic()-optimizer_started,
            optimizer_step=optimizer.global_step, optimizer_precision=precision,
            optimizer_threading=threading, update_seconds=time.monotonic()-started,
            maximum_generation_scoring_logratio=max(d['generation_scoring_max_logratio'] for d in details),
            initial_ratios=drift_summary)
    finally:
        optimizer.zero_grad()
        accumulator.buffers.clear()
