#!/usr/bin/env python3
"""llms.txt z bar text: time-relaxed (linear) bar -> fixed z > 1 (RA/20261102-zbar-flat). usage: patch_texts_zflat.py apply|revert FROM_EPOCH FILE...
Exact, reversible (revert puts the E514 linear-drop text back)."""
import os, sys
OLD = ("- **GREEN iff Delta > 0 AND z > z_bar (strict) AND cap-hit < 0.33 AND >= 100 drawn tasks scored (and >= 95 % of the draw).**\n"
       "  z_bar linear drop (from E514): z_bar = max(0.25, 1.0 - 0.75 * h / 12), h = hours since the last GREEN: 1.00 right after a\n"
       "  GREEN, 0.625 at 6 h, the 0.25 floor from 12 h; back to 1.00 after every GREEN. Clock = the candidate's TRAINING-RESULT time\n"
       "  minus the last GREEN's decided_at (both public in https://affine.io/gate.json entries[].timestamps), so gate latency does not\n"
       "  move the bar. The exact bar each turn was judged against: entries[].thresholds.z_min. The current bar, the formula and when\n"
       "  it reaches the floor: gate.json `z_bar` (also /gate/zbar.json, mining.json heldout.z_bar, the Gate z chart's z bar line).\n"
       "  E513-E513: steps (1.00, 0.50 from 6 h, 0.25 from 12 h). Before E513: z > 1.00.\n")
NEW = ("- **GREEN iff Delta > 0 AND z > 1 (strict) AND cap-hit < 0.33 AND >= 100 drawn tasks scored (and >= 95 % of the draw).**\n"
       "  The z bar is fixed at 1.00 (gate.json `z_bar`, mining.json heldout.z_bar); the bar each turn was judged against:\n"
       "  gate.json entries[].thresholds.z_min.\n"
       "  History: from E{E} the bar is fixed at z > 1; E513-E{P} used a time-relaxed bar.\n")
mode, e = sys.argv[1], sys.argv[2]; new = NEW.replace('{E}', e).replace('{P}', str(int(e) - 1))
for p in sys.argv[3:]:
    s = open(p).read(); a, b = (OLD, new) if mode == 'apply' else (new, OLD)
    if s.count(b) == 1 and s.count(a) == 0: print('%s %s: already' % (mode, p)); continue
    if s.count(a) != 1: raise SystemExit('%s: anchor count %d' % (p, s.count(a)))
    t = p + '.tmp.zf'; open(t, 'w').write(s.replace(a, b)); os.chmod(t, os.stat(p).st_mode & 0o777); os.replace(t, p); print(mode, p)
