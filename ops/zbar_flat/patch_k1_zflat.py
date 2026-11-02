#!/usr/bin/env python3
"""Gate z chart -> the flat z = 1 bar only (RA/20261102-zbar-flat). usage: patch_k1_zflat.py apply|revert K1JS
Exact anchored replacements (k1_zflat_pairs.json): each anchor must occur once (or the replacement is already there).
Leaves the rest of k1.js untouched, so concurrent edits elsewhere in the file survive."""
import json, os, sys
from pathlib import Path
mode, p = sys.argv[1], sys.argv[2]
pairs = json.load(open(Path(__file__).with_name('k1_zflat_pairs.json')))
s = open(p).read()
for a, b in pairs:
    if mode == 'revert': a, b = b, a
    if s.count(a) == 1: s = s.replace(a, b, 1)
    elif s.count(a) == 0 and s.count(b) >= 1: continue   # already
    else: raise SystemExit('%s: anchor count %d: %r' % (p, s.count(a), a[:90]))
t = p + '.tmp.zf'; open(t, 'w').write(s); os.chmod(t, os.stat(p).st_mode & 0o777); os.replace(t, p); print(mode, p)
