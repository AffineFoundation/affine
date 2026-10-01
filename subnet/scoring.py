"""Epoch uniqueness, with invalid submissions excluded before collision counting."""
from collections import Counter
from .protocol import sample_key

def score(reports):
    indices={miner:{sample_key(b) for b in report['accepted']} for miner,report in reports.items()}
    counts=Counter(index for values in indices.values() for index in values)
    points={miner:sum(counts[index]==1 for index in values) for miner,values in indices.items()}
    total=sum(points.values())
    provisional=any(any(o.get('fully_audited',True) is False for o in r.get('outcomes',[])) for r in reports.values())
    return dict(points=points,weights={m:p/total for m,p in points.items()} if total else {},total=total,provisional=provisional,score_basis='fully-audited-subset' if provisional else 'full-audit',unchecked_duplicate_claims_unresolved=provisional,duplicate_coverage='incomplete' if provisional else 'complete',assurance={m:r.get('assurance') for m,r in reports.items()})
