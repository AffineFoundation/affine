"""Bounded post-freeze work allocation and explicit reward penalties.

No credentials, blockchain writes, model execution, or implicit fraud inference.
Callers authenticate reports and freeze submission hashes before using this API.
"""
import hashlib
import math
import random

class InvalidSample(ValueError):
    """An explicit comparison proved submitted data incorrect, not an execution error."""

VERSION = 'bounded-random-v1'
DEFAULT_PENALTIES = dict(invalid_batch_multiplier=0.5, zero_epoch_after=0,
                         penalize_structural=False)


def integer(value, name, maximum=1000000):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(name)
    return value


def penalties(raw=None):
    result = dict(DEFAULT_PENALTIES)
    if raw is not None:
        if not isinstance(raw, dict) or set(raw) - set(result):
            raise ValueError('unknown penalty parameters')
        result.update(raw)
    multiplier = result['invalid_batch_multiplier']
    if type(multiplier) not in (int, float) or not math.isfinite(multiplier) or not 0 <= multiplier <= 1:
        raise ValueError('invalid batch multiplier')
    integer(result['zero_epoch_after'], 'zero epoch threshold')
    if type(result['penalize_structural']) is not bool:
        raise ValueError('structural penalty flag')
    return result


def validate(raw):
    fields = {'mode', 'version', 'epoch_budget', 'escalation_budget',
              'minimum_per_miner', 'maximum_per_miner', 'penalties', 'submission_counts'}
    if not isinstance(raw, dict) or set(raw) - fields or raw.get('mode') != 'sampled' or raw.get('version') != VERSION:
        raise ValueError('bounded audit policy')
    result = dict(raw)
    for key in ('epoch_budget', 'escalation_budget', 'minimum_per_miner', 'maximum_per_miner'):
        integer(result.get(key), key)
    if not 1 <= result['maximum_per_miner'] <= 10000 or result['minimum_per_miner'] > result['maximum_per_miner']:
        raise ValueError('per miner audit bounds')
    counts=result.get('submission_counts',{})
    if not isinstance(counts,dict) or len(counts)>4096:raise ValueError('submission audit allocations')
    for digest,count in counts.items():
        selection(0,count,'0'*64,digest)
        if count>result['maximum_per_miner']:raise ValueError('allocation exceeds per miner cap')
    result['penalties'] = penalties(result.get('penalties'))
    return result


def seeded(seed, domain):
    if not isinstance(seed, str) or len(seed) != 64:
        raise ValueError('post-freeze seed')
    try:
        if len(bytes.fromhex(seed)) != 32: raise ValueError('seed')
    except ValueError:
        raise ValueError('post-freeze seed') from None
    return random.Random(int.from_bytes(hashlib.sha256((seed + ':' + domain).encode()).digest(), 'big'))


def allocate(populations, policy, seed, *, escalation=False):
    """Fair minimum allocations followed by random remaining batch slots.

    Minimum allocations are a target, not a promise when capacity is too small.
    Populations may be conservative signed submission caps; unused slots are
    reported by workers and never counted as audits. Work is bounded by the registry size and configured audit slots.
    """
    policy = validate(policy)
    if not isinstance(populations, dict) or len(populations) > 4096:
        raise ValueError('audit population registry')
    limits = {}
    for miner, count in populations.items():
        if not isinstance(miner, str) or not miner:
            raise ValueError('miner identity')
        limits[miner] = min(integer(count, 'batch population', 10000), policy['maximum_per_miner'])
    counts = {m: 0 for m in sorted(limits)}
    remaining = policy['escalation_budget' if escalation else 'epoch_budget']
    rng = seeded(seed, 'escalation' if escalation else 'initial')
    eligible = [m for m in counts if limits[m]]
    # First spread the minimum allocation fairly, then sample remaining slots
    # proportionally to each miner's unallocated population.
    minimum=0 if escalation else policy['minimum_per_miner']
    for _ in range(minimum):
        if not remaining or not eligible:break
        rng.shuffle(eligible)
        for miner in eligible:
            if not remaining:break
            if counts[miner]<limits[miner]:
                counts[miner]+=1;remaining-=1
        eligible=[m for m in eligible if counts[m]<limits[m]]
    while remaining and eligible:
        residual=[limits[m]-counts[m] for m in eligible]
        pick=rng.randrange(sum(residual))
        for miner,slots in zip(eligible,residual):
            if pick<slots:
                counts[miner]+=1;remaining-=1
                if counts[miner]==limits[miner]:eligible.remove(miner)
                break
            pick-=slots
    return counts


def selection(total, count, seed, submission_hash):
    integer(total, 'batch population', 10000)
    integer(count, 'batch count', 10000)
    if not isinstance(submission_hash, str) or len(submission_hash) != 64:
        raise ValueError('frozen submission hash')
    try:
        if len(bytes.fromhex(submission_hash)) != 32: raise ValueError('hash')
    except ValueError:
        raise ValueError('frozen submission hash') from None
    # A prefix of a single permutation: escalated audits retain prior selections.
    values = list(range(total))
    seeded(seed, 'batches:' + submission_hash).shuffle(values)
    return sorted(values[:min(count, total)])


def penalty_count(report, policy):
    policy = penalties(policy)
    invalid = set()
    for outcome in report.get('outcomes', []):
        kind = outcome.get('failure_kind')
        if outcome.get('valid') is False and (
                kind == 'confirmed_invalid' and outcome.get('fully_audited') is True
                or kind == 'structural_invalid' and policy['penalize_structural']):
            batch = outcome.get('batch')
            if type(batch) is int and batch >= 0: invalid.add(batch)
    return len(invalid)


def reward_multiplier(report, policy):
    policy = penalties(policy)
    count = penalty_count(report, policy)
    zero = policy['zero_epoch_after']
    return (0.0 if zero and count >= zero else policy['invalid_batch_multiplier'] ** count), count


def escalation_allocations(populations, initial, policy, seed):
    """Charge repeated checks too: expanding c audits to c+e costs c+e slots."""
    policy=validate(policy)
    remaining=policy['escalation_budget']
    result={m:0 for m in sorted(populations)}
    order=list(result);seeded(seed,'expanded-jobs').shuffle(order)
    for miner in order:
        cap=min(integer(populations[miner],'escalation population',10000),policy['maximum_per_miner'])
        prior=integer(initial[miner],'initial allocation',10000)
        if cap<=prior or remaining<=prior:continue
        extra=min(cap-prior,remaining-prior)
        result[miner]=extra
        remaining-=prior+extra
    return result
