#!/usr/bin/env python3
"""Task-level cap rule text in llms.txt (RA/20261101-cap-task-level). usage: patch_texts_cap.py apply|revert FROM_EPOCH FILE... (exact, reversible)"""
import os, sys
OLD = ("- **GREEN iff Delta > 0 AND z > 1.00 (strict) AND estimated cap-hit on affine_math < 0.20 AND >= 100 drawn tasks scored\n"
       "  (and >= 95 % of the draw).** With N = 500")
NEW = ("- **GREEN iff Delta > 0 AND z > 1.00 (strict) AND cap-hit < 0.33 AND >= 100 drawn tasks scored (and >= 95 % of the draw).**\n"
       "  Cap-hit (task-level, from E{E}): the share of all the candidate's answers on the scored baseline tasks that hit the token\n"
       "  cap, every task counted alike: RED if >= 33% of the candidate's baseline answers hit the token cap (before E{E}: affine_math\n"
       "  cap-hit < 0.20). With N = 500")
mode, e = sys.argv[1], sys.argv[2]; new = NEW.replace('{E}', e)
for p in sys.argv[3:]:
    s = open(p).read(); a, b = (OLD, new) if mode == 'apply' else (new, OLD)
    if s.count(b) == 1 and s.count(a) == 0: print('%s %s: already' % (mode, p)); continue
    if s.count(a) != 1: raise SystemExit('%s: anchor count %d' % (p, s.count(a)))
    t = p + '.tmp.cap'; open(t, 'w').write(s.replace(a, b)); os.chmod(t, os.stat(p).st_mode & 0o777); os.replace(t, p); print(mode, p)
