"""Hourly contribution EMA; training/publication are deliberately not inputs."""
import math

VERSION = 'hourly-current-miner-assessment-v1'
HALF_LIFE_HOURS = 6
HISTORY_HOURS = 168
UNITS_PER_POINT = 1_000_000_000


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('finite nonnegative assessment input')
    return float(value)


def calculate(snapshots, committed_at, cutoff):
    """Reassess contribution history using current authenticated audit estimates.

    Smooth raw estimated contribution, then apply the current penalty once.
    Thus an adverse audit is not diluted by the EMA. Input snapshots are supplied
    by the authenticated evidence reader, never miner JSON or training results.
    """
    if type(cutoff) is not int or cutoff % 3600:
        raise ValueError('integral UTC assessment hour')
    latest, events, seen = {}, {}, set()
    for snap in sorted(snapshots, key=lambda s: (s['round'], s['epoch'])):
        epoch = snap['epoch']
        if epoch in seen or snap['cutoff'] != cutoff:
            raise ValueError('unique same-cutoff assessment snapshot')
        seen.add(epoch)
        at = number(committed_at[epoch])
        if at > cutoff:
            raise ValueError('future contribution')
        # A completed hour contains commitments in (previous cutoff, cutoff].
        hour = int(math.ceil(at / 3600)) * 3600
        for miner, details in snap['miners'].items():
            count = details['unique_eligible_batches']
            if type(count) is not int or count < 0:
                raise ValueError('eligible unique batch count')
            for field in ('validity_probability', 'reward_multiplier'):
                if number(details[field]) > 1:
                    raise ValueError('assessment factor above one')
            coverage = number(details.get('resolution_coverage_factor', 1))
            if coverage > 1:
                raise ValueError('resolution coverage above one')
            latest[miner] = dict(details)
            events.setdefault(hour, {})[miner] = events.setdefault(hour, {}).get(miner, 0) + count
    alpha = 1 - 2 ** (-1 / HALF_LIFE_HOURS)
    moving = {miner: 0. for miner in latest}
    first = cutoff - (HISTORY_HOURS - 1) * 3600
    for hour in range(first, cutoff + 1, 3600):
        for miner, details in latest.items():
            raw = events.get(hour, {}).get(miner, 0) * details['validity_probability'] * details.get('resolution_coverage_factor', 1.)
            moving[miner] = (1 - alpha) * moving[miner] + alpha * raw
    scores = {m: moving[m] * latest[m]['reward_multiplier'] for m in latest}
    total = sum(scores.values())
    return dict(version=VERSION, cutoff=cutoff, half_life_hours=HALF_LIFE_HOURS,
                hourly_alpha=alpha, history_hours=HISTORY_HOURS,
                smoothing_basis='estimated-valid-contribution-before-penalty',
                penalties_applied_after_smoothing=True,
                estimated_contribution_ema=moving, points=scores,
                weights={m: v / total if total else 0. for m, v in scores.items()},
                miner_estimates=latest, epochs=sorted(seen),
                training_completion_required=False,
                unaudited_samples_claimed_verified=False, chain_executed=False)


def fallback(previous, cutoff, reason, *, constraints=()):
    """An evidence outage preserves last assessment; inactivity is different."""
    if previous.get('version') != VERSION or previous['cutoff'] > cutoff:
        raise ValueError('last authenticated current assessment')
    result = dict(previous, cutoff=cutoff, evidence_cutoff=previous.get('evidence_cutoff', previous['cutoff']),
                  assessment_stale=True, evidence_error=reason, chain_executed=False)
    # Reusing historical contribution must not restore a later confirmed zero,
    # blacklist, or stronger penalty. Missing observations cannot erase evidence.
    scores = {m: number(v) for m, v in previous['points'].items()}
    estimates = {m: dict(v) for m, v in previous['miner_estimates'].items()}
    for current in sorted(constraints, key=lambda a: a['cutoff']):
        if current['cutoff'] < previous['cutoff']:
            continue
        for miner in scores:
            old = estimates[miner]
            new = current['miner_estimates'].get(miner)
            if new is None:
                continue
            for field in ('validity_probability', 'resolution_coverage_factor', 'reward_multiplier'):
                before = number(old.get(field, 1.))
                after = min(before, number(new.get(field, 1.)))
                if after > 1: raise ValueError('assessment factor above one')
                scores[miner] = scores[miner] * (after / before) if before else 0.
                old[field] = after
            old['blacklisted'] = bool(old.get('blacklisted') or new.get('blacklisted'))
            if old['blacklisted']: scores[miner] = 0.
    total = math.fsum(scores.values())
    result.update(points=scores, miner_estimates=estimates,
                  weights={m: v / total if total else 0. for m,v in scores.items()},
                  fallback_constraint_cutoffs=sorted({a['cutoff'] for a in constraints}))
    return result


def recipients(assessment, registrations):
    """Map cryptographic identity to current chain registration, never stale UID."""
    by_public = {}
    for hotkey, row in registrations.items():
        key = row['public_key']
        if key in by_public:
            raise ValueError('ambiguous registered public identity')
        by_public[key] = (hotkey, row)
    from subnet.chain import OWNER
    eligible, selected, excluded = {}, {}, []
    for miner, score in assessment['points'].items():
        score = number(score)
        if score <= 0: continue
        if miner not in by_public:
            excluded.append(miner); continue
        hotkey, row = by_public[miner]
        if hotkey == OWNER:
            excluded.append(miner); continue
        eligible[hotkey] = score
        selected[hotkey] = row
    # Relative positive shares, not absolute decayed EMA units. A tiny but valid
    # assessment must not disappear through integer underflow.
    largest = max(eligible.values(), default=1.)
    scaled = {h: score / largest for h,score in eligible.items()}
    total = math.fsum(scaled.values())
    points = {h: max(1, int(v / total * UNITS_PER_POINT)) for h,v in scaled.items()}
    return points, selected, sorted(excluded)
