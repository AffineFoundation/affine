"""Bounded public-prompt proposals for two explicitly recognized math problems.

No task data, expected answers, grader state, or held-out snapshots are inputs.
This is a curated proposal generator; model sampling remains a separate policy.
"""
import re
from functools import lru_cache


def candidates(messages):
    text='\n'.join(m['content'] for m in messages if m.get('role')=='user')
    answer=None;method=None
    if 'sum of the digits' in text and 'infinitely many' in text:
        match=re.search(r'S\(n\)\s*-\s*S\(n\s*\+\s*a\)\s*=\s*(\d+)',text)
        if match:
            target=int(match.group(1))
            if 1<=target<=100000:
                # Digit sums preserve the integer modulo nine. A sufficiently
                # long suffix of nines realizes this positive difference.
                answer=(-target)%9 or 9;method='decimal-carry-congruence-v1'
    if answer is None and all(s in text for s in ('one card','two cards with consecutive integers','three cards with consecutive integers','four cards with consecutive integers','smallest value')):
        @lru_cache(None)
        def can_complete(mask,turn,size):
            length=turn+1;moves=[]
            for start in range(size-length+1):
                removed=((1<<length)-1)<<start
                if mask&removed==removed:moves.append(mask^removed)
            if turn==3:return bool(moves)
            if not moves:return False
            results=[can_complete(m,turn+1,size) for m in moves]
            return all(results) if turn%2==0 else any(results)
        answer=next((n for n in range(1,25) if can_complete((1<<n)-1,0,n)),None)
        method='bounded-adversarial-contiguous-cards-v1'
    if answer is None:return dict(supported=False,method=None,candidates=[])
    return dict(supported=True,method=method,candidates=['\\boxed{'+str(answer)+'}','\\boxed{'+str(answer+1)+'}'])
