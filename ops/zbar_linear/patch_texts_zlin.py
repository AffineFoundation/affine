#!/usr/bin/env python3
"""z bar step-down -> linear drop text in llms.txt (RA/20261101-zbar-linear). usage: patch_texts_zlin.py apply|revert FROM_EPOCH FILE...
(exact, reversible; revert puts the E513 step-down text back)"""
import os, sys
OLD = ("  z_bar step-down (from E513): z_bar = 1.00, then 0.50 once 6 h and 0.25 once 12 h have passed since the last GREEN; back to\n"
       "  1.00 after every GREEN. Clock = the candidate's TRAINING-RESULT time minus the last GREEN's decided_at (both public in\n"
       "  https://affine.io/gate.json entries[].timestamps), so gate latency does not move the bar. The bar each turn was judged\n"
       "  against: entries[].thresholds.z_min. The current bar and when it next drops: gate.json `z_bar` (also /gate/zbar.json,\n"
       "  mining.json heldout.z_bar, the Gate z chart's z bar line). Before E513: z > 1.00.\n")
NEW = ("  z_bar linear drop (from E{E}): z_bar = max(0.25, 1.0 - 0.75 * h / 12), h = hours since the last GREEN: 1.00 right after a\n"
       "  GREEN, 0.625 at 6 h, the 0.25 floor from 12 h; back to 1.00 after every GREEN. Clock = the candidate's TRAINING-RESULT time\n"
       "  minus the last GREEN's decided_at (both public in https://affine.io/gate.json entries[].timestamps), so gate latency does not\n"
       "  move the bar. The exact bar each turn was judged against: entries[].thresholds.z_min. The current bar, the formula and when\n"
       "  it reaches the floor: gate.json `z_bar` (also /gate/zbar.json, mining.json heldout.z_bar, the Gate z chart's z bar line).\n"
       "  E513-E{P}: steps (1.00, 0.50 from 6 h, 0.25 from 12 h). Before E513: z > 1.00.\n")
mode, e = sys.argv[1], sys.argv[2]; new = NEW.replace('{E}', e).replace('{P}', str(int(e) - 1))
for p in sys.argv[3:]:
    s = open(p).read(); a, b = (OLD, new) if mode == 'apply' else (new, OLD)
    if s.count(b) == 1 and s.count(a) == 0: print('%s %s: already' % (mode, p)); continue
    if s.count(a) != 1: raise SystemExit('%s: anchor count %d' % (p, s.count(a)))
    t = p + '.tmp.zl'; open(t, 'w').write(s.replace(a, b)); os.chmod(t, os.stat(p).st_mode & 0o777); os.replace(t, p); print(mode, p)
