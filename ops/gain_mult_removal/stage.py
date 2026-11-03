"""Stage the gain-multiplier removal (Jacob t1150u). Reads live files, writes stage/. No live writes."""
import json, re, shutil, sys
from pathlib import Path
H = Path('/home/const/subnet120-rewrite'); RA = H / 'state/root-audits'; R = RA / '20261103-gain-mult-removal'; S = R / 'stage'
G = RA / '20261102-gain-mult'; PRE = G / 'backups/pre-20261102T224102Z'; W = Path('/home/const/subnet120/affine/website')
S.mkdir(exist_ok=True)
# k1_service + controller: the pre-multiplier files (live == pre + gain-mult diff exactly, checked)
shutil.copy2(PRE / 'k1_service.py', S / 'k1_service.py')
shutil.copy2(PRE / 'controller.py', S / 'controller.py')
# controller.json: drop parent_hold only (keep the current pod id)
c = json.loads((RA / '20261101-prior-table/controller.json').read_text()); c.pop('parent_hold', None)
(S / 'controller.json').write_text(json.dumps(c, indent=1))
# training_params: remove the two gain-mult hunks
t = (RA / '20261026-promotion-gate/code/training_params.py').read_text()
a = "                  transparency=TRANSPARENCY,\n                  king_gain_multiplier=_gain_mult_public())   # gain-mult 20261102"
assert t.count(a) == 1; t = t.replace(a, "                  transparency=TRANSPARENCY)")
b = ("def _gain_mult_public():   # gain-mult 20261102\n"
     "    try: return json.loads((RA / '20261102-gain-mult/state/GAIN-MULT.public.json').read_bytes())\n"
     "    except Exception: return None\n\n\n")
assert t.count(b) == 1; t = t.replace(b, '')
assert 'gain_mult' not in t and 'gain-mult' not in t
(S / 'training_params.py').write_text(t)
# llms.txt: drop the section-8 gain multiplier bullet
l = (W / 'llms.txt').read_text()
m = re.search(r"- King gain multiplier <!-- gain-mult-20261102 -->.*?Per-task data stays private\.\n", l, re.S)
assert m; l = l[:m.start()] + l[m.end():]
l = re.sub(r"Last updated: [0-9-]+ [0-9:]+ CR \(UTC-6\)\.", "Last updated: %s CR (UTC-6)." % sys.argv[1], l, count=1)
assert 'gain_m' not in l and 'multiplier' not in l.lower()
(S / 'llms.txt').write_text(l)
# caps tools: drop the gain-mult regime + parent-hold steps / consumer
ca = (H / 'tools/caps/caps_apply.py').read_text(); (S / 'caps_apply.py').write_text(ca)
co = (H / 'tools/caps/consumers.py').read_text(); (S / 'consumers.py').write_text(co)
print('staged', sorted(p.name for p in S.iterdir()))
