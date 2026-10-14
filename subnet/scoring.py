"""Epoch uniqueness, with invalid submissions excluded before collision counting.

First-claim-wins (``RULE``), enabled when the caller passes the frozen R2
``receipts`` for the epoch. When several miners hold an accepted claim on the
same task (``sample_key``), the claim whose bytes reached Cloudflare R2 first
wins the point and the others get nothing for that task:

* claim time is the per-artifact R2 ``LastModified`` (``received_at`` of the
  receipt artifact whose ``batch_sha256`` matches the accepted batch), falling
  back to the miner's commitment/receipt ``received_at``;
* timestamped claims beat untimestamped claims;
* equal earliest timestamps (R2 has one-second precision) are broken by the
  lowest ``sha256(miner_hotkey + repr(key))``;
* if no claim on a contested task has a timestamp, nobody is paid for it.

Without ``receipts`` the original zero-for-all rule applies unchanged, so
historical score documents (which carry no ``claim_order_rule``) still
recompute byte-for-byte.
"""
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from .protocol import sample_key

RULE = 'earliest-r2-received_at-v1'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _time(value):
    if type(value) in (int, float) and math.isfinite(value):
        return float(value)
    return None


def claim_time(receipt, batch):
    """R2 arrival time for one accepted batch; ``None`` when unknown."""
    if not isinstance(receipt, dict):
        return None
    digest = hashlib.sha256(_canonical(batch)).hexdigest()
    times = [_time(a.get('received_at')) for a in receipt.get('artifacts') or []
             if isinstance(a, dict) and a.get('batch_sha256') == digest]
    times = [t for t in times if t is not None]
    if times:
        return min(times)
    return _time(receipt.get('received_at'))


def tie_break(miner, key):
    return hashlib.sha256((str(miner) + repr(tuple(key))).encode()).hexdigest()


def _rank(when):
    # Timestamped claims sort before untimestamped ones.
    return (0, when) if when is not None else (1, 0.0)


def resolve_claims(reports, receipts=None):
    """Return ``{key: (winner_or_None, received_at_or_None, claimant_count)}``."""
    claims = defaultdict(dict)
    for miner, report in reports.items():
        receipt = (receipts or {}).get(miner)
        for batch in report['accepted']:
            key = sample_key(batch)
            when = claim_time(receipt, batch) if receipts is not None else None
            if miner not in claims[key] or _rank(when) < _rank(claims[key][miner]):
                claims[key][miner] = when
    resolved = {}
    for key, owners in claims.items():
        if len(owners) == 1:
            (miner, when), = owners.items()
            resolved[key] = (miner, when, 1)
            continue
        timed = [(when, tie_break(miner, key), miner) for miner, when in owners.items() if when is not None]
        if not timed:
            resolved[key] = (None, None, len(owners))
            continue
        when, _, miner = min(timed)
        resolved[key] = (miner, when, len(owners))
    return resolved


def unique_points(reports, receipts=None):
    points = {miner: 0 for miner in reports}
    for winner, _, _ in resolve_claims(reports, receipts).values():
        if winner is not None:
            points[winner] += 1
    return points


def contested_claims(reports, receipts):
    """Publishable record of every task claimed by more than one miner."""
    return [dict(key=list(key), winner=winner, received_at=when, claimants=count)
            for key, (winner, when, count) in sorted(resolve_claims(reports, receipts).items(), key=lambda item: repr(item[0]))
            if count > 1]


def _claim_fields(reports, receipts):
    if receipts is None:
        return {}
    return dict(claim_order_rule=RULE, contested_claims=contested_claims(reports, receipts))

def adjusted_point_fractions(reports, penalty_policy, receipts=None):
    """Exact audit-only reward arithmetic shared by previews and payout generation.

    Authentication belongs to the caller. Fractions keep hourly aggregation from
    rounding each epoch separately; this function never reads chain or model state.
    """
    from .audit_policy import penalties, penalty_count
    policy=penalties(penalty_policy)
    points=unique_points(reports,receipts)
    multiplier=Fraction(str(policy['invalid_batch_multiplier']))
    adjustments={}
    adjusted={}
    for miner,report in reports.items():
        count=penalty_count(report,policy)
        factor=Fraction(0) if policy['zero_epoch_after'] and count>=policy['zero_epoch_after'] else multiplier**count
        adjustments[miner]=(factor,count)
        adjusted[miner]=points[miner]*factor
    return points,adjusted,adjustments

def score(reports, penalty_policy=None, receipts=None):
    points=unique_points(reports,receipts)
    total=sum(points.values())
    provisional=any(any(o.get('valid') is None or o.get('failure_kind')=='verification_error' for o in r.get('outcomes',[])) for r in reports.values())
    if penalty_policy is not None:
        from .audit_policy import penalties
        policy=penalties(penalty_policy)
        points,fractions,adjustments=adjusted_point_fractions(reports,policy,receipts)
        adjusted={m:float(p) for m,p in fractions.items()}
        denominator=sum(fractions.values())
        return dict(points=points,adjusted_points=adjusted,total=total,
                    weights={m:float(p/denominator) for m,p in fractions.items()} if denominator else {},
                    penalty_policy=policy,penalties={m:dict(multiplier=float(v[0]),confirmed_invalid_batches=v[1]) for m,v in adjustments.items()},
                    provisional=provisional,score_basis='fully-audited-subset' if provisional else 'full-audit',
                    unchecked_duplicate_claims_unresolved=provisional,
                    duplicate_coverage='incomplete' if provisional else 'complete',
                    assurance={m:r.get('assurance') for m,r in reports.items()},**_claim_fields(reports,receipts))
    return dict(points=points,weights={m:p/total for m,p in points.items()} if total else {},total=total,provisional=provisional,score_basis='fully-audited-subset' if provisional else 'full-audit',unchecked_duplicate_claims_unresolved=provisional,duplicate_coverage='incomplete' if provisional else 'complete',assurance={m:r.get('assurance') for m,r in reports.items()},**_claim_fields(reports,receipts))
