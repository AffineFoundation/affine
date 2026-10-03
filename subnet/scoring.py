"""Epoch uniqueness, with invalid submissions excluded before collision counting."""
from collections import Counter
from fractions import Fraction
from .protocol import sample_key

def unique_points(reports):
    indices={miner:{sample_key(b) for b in report['accepted']} for miner,report in reports.items()}
    counts=Counter(index for values in indices.values() for index in values)
    return {miner:sum(counts[index]==1 for index in values) for miner,values in indices.items()}

def adjusted_point_fractions(reports, penalty_policy):
    """Exact audit-only reward arithmetic shared by previews and payout generation.

    Authentication belongs to the caller. Fractions keep hourly aggregation from
    rounding each epoch separately; this function never reads chain or model state.
    """
    from .audit_policy import penalties, penalty_count
    policy=penalties(penalty_policy)
    points=unique_points(reports)
    multiplier=Fraction(str(policy['invalid_batch_multiplier']))
    adjustments={}
    adjusted={}
    for miner,report in reports.items():
        count=penalty_count(report,policy)
        factor=Fraction(0) if policy['zero_epoch_after'] and count>=policy['zero_epoch_after'] else multiplier**count
        adjustments[miner]=(factor,count)
        adjusted[miner]=points[miner]*factor
    return points,adjusted,adjustments

def score(reports, penalty_policy=None):
    points=unique_points(reports)
    total=sum(points.values())
    provisional=any(any(o.get('valid') is None or o.get('failure_kind')=='verification_error' for o in r.get('outcomes',[])) for r in reports.values())
    if penalty_policy is not None:
        from .audit_policy import penalties
        policy=penalties(penalty_policy)
        points,fractions,adjustments=adjusted_point_fractions(reports,policy)
        adjusted={m:float(p) for m,p in fractions.items()}
        denominator=sum(fractions.values())
        return dict(points=points,adjusted_points=adjusted,total=total,
                    weights={m:float(p/denominator) for m,p in fractions.items()} if denominator else {},
                    penalty_policy=policy,penalties={m:dict(multiplier=float(v[0]),confirmed_invalid_batches=v[1]) for m,v in adjustments.items()},
                    provisional=provisional,score_basis='fully-audited-subset' if provisional else 'full-audit',
                    unchecked_duplicate_claims_unresolved=provisional,
                    duplicate_coverage='incomplete' if provisional else 'complete',
                    assurance={m:r.get('assurance') for m,r in reports.items()})
    return dict(points=points,weights={m:p/total for m,p in points.items()} if total else {},total=total,provisional=provisional,score_basis='fully-audited-subset' if provisional else 'full-audit',unchecked_duplicate_claims_unresolved=provisional,duplicate_coverage='incomplete' if provisional else 'complete',assurance={m:r.get('assurance') for m,r in reports.items()})
