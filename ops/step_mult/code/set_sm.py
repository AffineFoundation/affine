#!/usr/bin/env python3
"""RA/20261102-step-mult: set the accepted step_mult values (env SM_ALLOWED, e.g. 1,2,4) in staged ff_format.py / affine_batch.py copies (in place).
usage: SM_ALLOWED=1,2,4 set_sm.py FILE..."""
import os, sys
S = tuple(sorted({int(x) for x in os.environ['SM_ALLOWED'].split(',')})); assert S[0] == 1 and set(S) <= set(range(1, 9)); N = max(S); L = ', '.join(map(str, S))
for p in sys.argv[1:]:
    s = open(p).read(); o = s
    a = 'STEP_MULT_ALLOWED = (1, 2, 3, 4, 5, 6, 7, 8)'; assert s.count(a) == 1, (p, a)
    s = s.replace(a, 'STEP_MULT_ALLOWED = %r' % (S,))
    m = "'train.step_mult must be an integer 1..8 (got %s)'"; assert s.count(m) == 1, (p, m)
    s = s.replace(m, "'train.step_mult must be %s (got %%s)'" % ('one of the integers %s' % L if N > 1 else '1 (only 1 is accepted for now)'))
    s = s.replace('optional INTEGER 1..8', 'optional INTEGER, one of %s' % L).replace('step_mult an integer 1..8', 'step_mult one of %s' % L)
    open(p, 'w').write(s); print(p, 'step_mult', S, 'changed' if s != o else 'unchanged')
