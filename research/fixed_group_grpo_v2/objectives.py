"""Auditable differentiable objectives, independent of model/remote operations.

GRPO uses the actual captured behavior LP denominator, never current.detach()
as a substitute. KL is the original sampled nonnegative k3 expression, with its
current-policy gradient intact. It is a sampled fixed-data surrogate, not an
exact full-vocabulary forward KL or detached KL reward shaping.
"""
from dataclasses import dataclass
import math
import torch
from .core import require, validate_group


@dataclass(frozen=True)
class Objective:
    mode: str = 'grpo'
    clip_epsilon: float = .2
    kl_beta: float = .01
    advantage_epsilon: float = 1e-4
    normalization: str = 'sequence_mean'
    fixed_token_divisor: int = 4096
    preference_beta: float = .1

    def validate(self):
        require(self.mode in ('grpo', 'group_preference'), 'declared objective')
        require(0 < self.clip_epsilon < 1 and math.isfinite(self.clip_epsilon), 'ratio clipping epsilon')
        require(math.isfinite(self.kl_beta) and self.kl_beta >= 0, 'nonnegative finite KL beta')
        require(math.isfinite(self.advantage_epsilon) and self.advantage_epsilon > 0, 'advantage epsilon')
        require(self.normalization in ('sequence_mean', 'fixed_tokens'), 'declared length normalization')
        require(type(self.fixed_token_divisor) is int and self.fixed_token_divisor > 0, 'constant length divisor')
        require(math.isfinite(self.preference_beta) and self.preference_beta > 0, 'preference beta')


def group_advantages(group, *, group_size, epsilon=1e-4):
    validate_group(group, group_size)
    rewards = torch.tensor([float(r.outcome == 'correct') for r in group.rollouts], dtype=torch.float64)
    std = rewards.std(unbiased=False)
    advantage = (rewards - rewards.mean()) / (std + epsilon)
    # Unresolved rows stay in original group statistics, but their entire loss
    # is masked. They are never recast as completed wrong responses.
    return advantage.tolist(), dict(rewards=rewards.tolist(), mean=float(rewards.mean()),
        population_std=float(std), unresolved_mask=[r.outcome == 'unresolved' for r in group.rollouts])


def grpo_trajectory_loss(current, old, reference, *, advantage, objective, task_count,
                         group_size, active=True):
    objective.validate()
    require(current.ndim == 1 and current.numel() > 0 and current.requires_grad, 'differentiable generated-token LP')
    old = torch.as_tensor(old, device=current.device, dtype=current.dtype).detach()
    reference = torch.as_tensor(reference, device=current.device, dtype=current.dtype).detach()
    require(old.shape == current.shape == reference.shape, 'all original generated tokens')
    require(all(bool(torch.isfinite(x).all()) and bool((x <= 0).all()) for x in (current, old, reference)), 'finite nonpositive LP')
    require(math.isfinite(advantage) and task_count > 0 and group_size > 1, 'valid complete group weighting')
    delta = current - old
    ratio = delta.exp()
    reference_delta = reference - current
    # expm1 is stable near zero. No detach on current; full autograd path stays.
    kl = torch.expm1(reference_delta) - reference_delta
    require(bool(torch.isfinite(ratio).all()) and bool(torch.isfinite(kl).all()), 'finite ratio and KL; abort instead of clamping invalid numerics')
    surrogate = torch.minimum(ratio * advantage,
                              ratio.clamp(1-objective.clip_epsilon, 1+objective.clip_epsilon) * advantage)
    divisor = current.numel() if objective.normalization == 'sequence_mean' else objective.fixed_token_divisor
    value = (-surrogate + objective.kl_beta * kl).sum() / (task_count * group_size * divisor)
    if not active:
        value = current.sum() * 0.
    diagnostic = dict(loss=float(value.detach()), advantage=float(advantage), active=bool(active),
        mean_kl=float(kl.detach().mean()), mean_ratio=float(ratio.detach().mean()),
        max_abs_token_logratio=float(delta.detach().abs().max()),
        clipped_fraction=float(((ratio < 1-objective.clip_epsilon) | (ratio > 1+objective.clip_epsilon)).float().mean()),
        token_count=current.numel(), policy_term=float((-surrogate).detach().mean()),
        loss_divisor=task_count*group_size*divisor)
    return value, diagnostic


def preference_coefficients(group, current_means, old_means, *, group_size, beta, task_count):
    """Exact derivative of all positive×completed-wrong pairwise logistic losses.

    Scalar prepass supplies means without model graphs. Return dL/d(mean LP) for
    each response; a second unchanged-model pass supplies one model graph at a
    time. This gives the exact first derivative without holding 8 long graphs.
    The reference is this update's old-policy score, reset each update.
    """
    validate_group(group, group_size)
    require(len(current_means) == len(old_means) == group_size and task_count > 0, 'complete preference group')
    x = torch.tensor(current_means, dtype=torch.float64, requires_grad=True)
    ref = torch.tensor(old_means, dtype=torch.float64)
    positives = [i for i, r in enumerate(group.rollouts) if r.outcome == 'correct']
    negatives = [i for i, r in enumerate(group.rollouts) if r.outcome == 'wrong']
    require(positives and negatives, 'mixed completed group')
    losses = [-torch.nn.functional.logsigmoid(beta * ((x[i]-x[j]) - (ref[i]-ref[j])))
              for i in positives for j in negatives]
    loss = torch.stack(losses).mean() / task_count
    gradient, = torch.autograd.grad(loss, x)
    return gradient.tolist(), dict(loss=float(loss.detach()), positive_count=len(positives),
                                  negative_count=len(negatives), comparison_count=len(losses))
