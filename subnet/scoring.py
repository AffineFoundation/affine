"""Epoch uniqueness, with invalid submissions excluded before collision counting."""
from collections import Counter
from .protocol import sample_key

def score(reports, penalty_policy=None):
    indices={miner:{sample_key(b) for b in report['accepted']} for miner,report in reports.items()}
    counts=Counter(index for values in indices.values() for index in values)
    points={miner:sum(counts[index]==1 for index in values) for miner,values in indices.items()}
    total=sum(points.values())
    provisional=any(any(o.get('valid') is None or o.get('failure_kind')=='verification_error' for o in r.get('outcomes',[])) for r in reports.values())
    if penalty_policy is not None:
        from .audit_policy import penalties, reward_multiplier
        policy=penalties(penalty_policy)
        adjustments={m:reward_multiplier(r,policy) for m,r in reports.items()}
        adjusted={m:points[m]*adjustments[m][0] for m in points}
        denominator=sum(adjusted.values())
        return dict(points=points,adjusted_points=adjusted,total=total,
                    weights={m:p/denominator for m,p in adjusted.items()} if denominator else {},
                    penalty_policy=policy,penalties={m:dict(multiplier=v[0],confirmed_invalid_batches=v[1]) for m,v in adjustments.items()},
                    provisional=provisional,score_basis='fully-audited-subset' if provisional else 'full-audit',
                    unchecked_duplicate_claims_unresolved=provisional,
                    duplicate_coverage='incomplete' if provisional else 'complete',
                    assurance={m:r.get('assurance') for m,r in reports.items()})
    return dict(points=points,weights={m:p/total for m,p in points.items()} if total else {},total=total,provisional=provisional,score_basis='fully-audited-subset' if provisional else 'full-audit',unchecked_duplicate_claims_unresolved=provisional,duplicate_coverage='incomplete' if provisional else 'complete',assurance={m:r.get('assurance') for m,r in reports.items()})
