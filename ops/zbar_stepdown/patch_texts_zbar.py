#!/usr/bin/env python3
"""z bar step-down text in llms.txt (RA/20261101-zbar-stepdown). usage: patch_texts_zbar.py apply|revert FROM_EPOCH FILE... (exact, reversible)"""
import os, sys
OLD = "- **GREEN iff Delta > 0 AND z > 1.00 (strict) AND cap-hit < 0.33 AND >= 100 drawn tasks scored (and >= 95 % of the draw).**\n"
NEW = ("- **GREEN iff Delta > 0 AND z > z_bar (strict) AND cap-hit < 0.33 AND >= 100 drawn tasks scored (and >= 95 % of the draw).**\n"
       "  z_bar step-down (from E{E}): z_bar = 1.00, then 0.50 once 6 h and 0.25 once 12 h have passed since the last GREEN; back to\n"
       "  1.00 after every GREEN. Clock = the candidate's TRAINING-RESULT time minus the last GREEN's decided_at (both public in\n"
       "  https://affine.io/gate.json entries[].timestamps), so gate latency does not move the bar. The bar each turn was judged\n"
       "  against: entries[].thresholds.z_min. The current bar and when it next drops: gate.json `z_bar` (also /gate/zbar.json,\n"
       "  mining.json heldout.z_bar, the Gate z chart's z bar line). Before E{E}: z > 1.00.\n")
mode, e = sys.argv[1], sys.argv[2]; new = NEW.replace('{E}', e)
for p in sys.argv[3:]:
    s = open(p).read(); a, b = (OLD, new) if mode == 'apply' else (new, OLD)
    if s.count(b) == 1 and s.count(a) == 0: print('%s %s: already' % (mode, p)); continue
    if s.count(a) != 1: raise SystemExit('%s: anchor count %d' % (p, s.count(a)))
    t = p + '.tmp.zb'; open(t, 'w').write(s.replace(a, b)); os.chmod(t, os.stat(p).st_mode & 0o777); os.replace(t, p); print(mode, p)
