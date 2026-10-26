"""Pure CPU guard over the entire pre-update training population.

Caller must supply ALL intended loss-bearing prepass rows before any backward or
optimizer step. This module does not authenticate behavior LP, validate complete
groups, recompute advantages, select data, or verify historical generation. It
never edits, replaces or clips the supplied likelihoods. Inactive rows may be
included for diagnostics but do not dilute the training-population gate.
"""
from dataclasses import asdict, dataclass
import math
from numbers import Real


class RatioInputError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise RatioInputError(message)


def number(value, label):
    require(isinstance(value, Real) and not isinstance(value, bool), label)
    require(math.isfinite(value), label + ' must be finite')
    return float(value)


@dataclass(frozen=True)
class Thresholds:
    max_abs_logratio: float = 1.0
    mean_abs_logratio: float = .02
    p99_abs_logratio: float = .15
    pooled_ppo_clip_fraction: float = .01
    max_per_trajectory_clip_fraction: float = .05
    clip_epsilon: float = .2

    def validate(self):
        for name, value in asdict(self).items():
            number(value, name)
            require(value >= 0., name + ' must be nonnegative')
        require(0. < self.clip_epsilon < 1., 'PPO epsilon in (0,1)')
        require(self.pooled_ppo_clip_fraction <= 1.
                and self.max_per_trajectory_clip_fraction <= 1., 'fractions in [0,1]')
        require(self.mean_abs_logratio <= self.max_abs_logratio
                and self.p99_abs_logratio <= self.max_abs_logratio,
                'aggregate limits do not exceed absolute ceiling')


@dataclass(frozen=True)
class Trajectory:
    task_id: str
    nonce: int
    outcome: str
    advantage: float
    behavior_logprobs: tuple
    current_logprobs: tuple
    active: bool = True


def quantile(values, fraction):
    """Linear interpolation, matching torch.quantile's default definition."""
    ordered = sorted(values)
    position = (len(ordered)-1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high]-ordered[low]) * (position-low)


def stats(deltas, lower, upper):
    absolute = [abs(x) for x in deltas]
    low = sum(x < lower for x in deltas)
    high = sum(x > upper for x in deltas)
    return dict(tokens=len(deltas), signed_mean=math.fsum(deltas)/len(deltas),
                signed_p01=quantile(deltas, .01), signed_p50=quantile(deltas, .5),
                signed_p99=quantile(deltas, .99), signed_min=min(deltas),
                signed_max=max(deltas), mean_abs_logratio=math.fsum(absolute)/len(deltas),
                p99_abs_logratio=quantile(absolute, .99), max_abs_logratio=max(absolute),
                below_ppo_band_tokens=low, above_ppo_band_tokens=high,
                outside_ppo_band_tokens=low+high,
                pooled_ppo_clip_fraction=(low+high)/len(deltas))


def assess(rows, thresholds=Thresholds()):
    """Return diagnostics + passed/failures; malformed inputs raise, never pass.

    Bounds are inclusive. PPO 'outside band' is NOT the same as a clipped
    objective gradient; both are reported. No epsilon/fudge is silently added.
    No exponentiation is needed, so a finite extreme log-ratio cannot overflow
    this diagnostic before its absolute-ceiling failure is recorded.
    """
    require(isinstance(thresholds, Thresholds), 'typed thresholds required')
    thresholds.validate()
    rows = tuple(rows)
    require(bool(rows), 'nonempty complete prepass population')
    lower, upper = math.log1p(-thresholds.clip_epsilon), math.log1p(thresholds.clip_epsilon)
    seen, pooled, reports, active_reports = set(), [], [], []
    active_clips, nonzero_tokens, weighted_active, weighted_total = 0, 0, [], []
    for row in rows:
        require(isinstance(row, Trajectory), 'typed trajectory required')
        require(isinstance(row.task_id, str) and bool(row.task_id), 'nonempty task identity')
        require(type(row.nonce) is int and row.nonce >= 0, 'nonnegative integer nonce')
        require((row.task_id, row.nonce) not in seen, 'duplicate trajectory identity')
        seen.add((row.task_id, row.nonce))
        require(row.outcome in ('correct', 'wrong', 'unresolved'), 'known outcome')
        require(type(row.active) is bool, 'explicit boolean loss mask')
        require(not (row.active and row.outcome == 'unresolved'), 'unresolved cannot bear loss')
        advantage = number(row.advantage, 'advantage')
        old, new = tuple(row.behavior_logprobs), tuple(row.current_logprobs)
        require(len(old) == len(new) > 0, 'nonempty aligned selected-token arrays')
        old = tuple(number(x, 'behavior logprob') for x in old)
        new = tuple(number(x, 'current logprob') for x in new)
        require(all(x <= 0 for x in old+new), 'log probabilities must be nonpositive')
        deltas = [b-a for a, b in zip(old, new)]
        require(all(math.isfinite(x) for x in deltas), 'finite subtraction')
        report = stats(deltas, lower, upper)
        actual_clips = sum((advantage > 0. and d > upper)
                           or (advantage < 0. and d < lower) for d in deltas)
        report.update(task_id=row.task_id, nonce=row.nonce, outcome=row.outcome,
                      active=row.active, advantage=advantage,
                      sign_aware_clipped_tokens=actual_clips if row.active else 0,
                      sign_aware_clipped_fraction=actual_clips/len(deltas)
                      if row.active and advantage != 0 else None)
        reports.append(report)
        if row.active:
            pooled.extend(deltas)
            active_reports.append(report)
            if advantage != 0.:
                active_clips += actual_clips
                nonzero_tokens += len(deltas)
                weighted_total.append(abs(advantage))
                weighted_active.append(abs(advantage)*actual_clips/len(deltas))
    require(bool(pooled), 'at least one active training trajectory required')
    aggregate = stats(pooled, lower, upper)
    aggregate['max_per_trajectory_clip_fraction'] = max(
        r['pooled_ppo_clip_fraction'] for r in active_reports)
    aggregate['nonzero_advantage_tokens'] = nonzero_tokens
    aggregate['sign_aware_clipped_tokens'] = active_clips
    aggregate['sign_aware_clipped_fraction'] = active_clips/nonzero_tokens if nonzero_tokens else None
    weight = math.fsum(weighted_total)
    aggregate['sequence_normalized_advantage_mass_clipped_fraction'] = (
        math.fsum(weighted_active)/weight if weight else None)
    failures = [dict(metric=name, measured=aggregate[name], limit=limit)
                for name, limit in asdict(thresholds).items()
                if name != 'clip_epsilon' and aggregate[name] > limit]
    return dict(version='prospective-group-ratio-usability-v1', passed=not failures,
                failures=failures, thresholds=asdict(thresholds),
                population='all caller-declared active pre-update training tokens',
                total_trajectories=len(rows), active_trajectories=len(active_reports),
                inactive_trajectories=len(rows)-len(active_reports),
                aggregate=aggregate, trajectories=reports,
                behavior_logprobs_modified=False, proof_acceptance=False)
