"""First-claim-wins for contested tasks (``earliest-r2-received_at-v1``).

A task is contested when more than one miner holds an eligible claim on it.
The claim whose artifact reached Cloudflare R2 first (``LastModified`` as
``received_at``) wins; timestamped claims beat untimestamped ones; same-second
ties go to the lowest ``sha256(miner + repr(key))``; if no claim carries a
timestamp nobody wins (the historical zero-for-all rule).

Epochs opt in through the signed opening manifest field ``claim_order_rule``.
Without it every caller keeps the historical rule byte-for-byte.
"""
import hashlib
import math

RULE = 'earliest-r2-received_at-v1'
FIELD = 'claim_order_rule'


def enabled(manifest):
    value = manifest.get(FIELD) if isinstance(manifest, dict) else None
    if value not in (None, RULE):
        raise ValueError('unknown claim order rule')
    return value == RULE


def time_or_none(value):
    if type(value) in (int, float) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def tie_break(miner, key):
    return hashlib.sha256((str(miner) + repr(tuple(key))).encode()).hexdigest()


def resolve(claims):
    """``claims``: iterable of ``(miner, key, received_at_or_None)``.

    Returns ``{key: (winner_or_None, received_at_or_None, claimant_count)}``.
    A miner's earliest timestamp on a key represents that miner.
    """
    owners = {}
    for miner, key, when in claims:
        key = tuple(key); when = time_or_none(when); seen = owners.setdefault(key, {})
        if miner not in seen or (when is not None and (seen[miner] is None or when < seen[miner])):
            seen[miner] = when
    result = {}
    for key, by_miner in owners.items():
        if len(by_miner) == 1:
            (miner, when), = by_miner.items(); result[key] = (miner, when, 1); continue
        timed = [(when, tie_break(miner, key), miner) for miner, when in by_miner.items() if when is not None]
        if not timed:
            result[key] = (None, None, len(by_miner)); continue
        when, _, miner = min(timed); result[key] = (miner, when, len(by_miner))
    return result


def contested(resolved):
    """Publishable list of contested tasks, deterministic order."""
    return [dict(key=list(key), winner=winner, received_at=when, claimants=count)
            for key, (winner, when, count) in sorted(resolved.items(), key=lambda item: repr(item[0])) if count > 1]


def winners(contested_claims):
    """Validate a published ``contested_claims`` list and map key -> winner."""
    if type(contested_claims) is not list:
        raise ValueError('contested claims list')
    result = {}
    for row in contested_claims:
        if type(row) is not dict or set(row) != {'key', 'winner', 'received_at', 'claimants'} or type(row['key']) is not list:
            raise ValueError('exact contested claim')
        key = tuple(row['key'])
        if key in result or type(row['claimants']) is not int or row['claimants'] < 2:
            raise ValueError('unique contested claim')
        result[key] = row['winner']
    return result
