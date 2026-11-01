#!/usr/bin/env python3
"""usage: sanitize_gate.py LIVE_PREV REPO_PREV LIVE_NOW OUT - learns the line substitutions the repo copy applies to the live
gate_service.py (host paths, probe SSH host/port/known_hosts -> placeholders) from the previous pair, applies them to the live
file now and refuses if a learned line is missing or a sensitive string survives."""
import difflib, re, sys
lp, rp, ln, out = sys.argv[1:5]
a, b, n = open(lp).read().splitlines(True), open(rp).read().splitlines(True), open(ln).read().splitlines(True)
sub = {}
for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
    if tag == 'equal': continue
    assert tag == 'replace' and i2 - i1 == j2 - j1, (tag, a[i1:i2], b[j1:j2])
    for x, y in zip(a[i1:i2], b[j1:j2]): sub[x] = y
miss = [k for k in sub if k not in n]
if miss: sys.exit('learned lines missing in the live file: %r' % miss[:3])
res = [sub.get(x, x) for x in n]
txt = ''.join(res)
bad = [m.group(0) for m in re.finditer(r'/home/const|\b\d{1,3}(\.\d{1,3}){3}\b|known_hosts/[0-9a-f-]{8,}', txt)]
if bad: sys.exit('sensitive strings left: %r' % bad[:5])
open(out, 'w').write(txt); print('sanitized %d lines -> %s' % (len(sub), out))
