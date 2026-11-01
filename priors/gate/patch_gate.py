#!/usr/bin/env python3
"""Install / remove the prior-v1 baseline hooks (RA/20261101-baseline-switch). Anchor-based, idempotent, exact reverse.
usage: patch_gate.py apply|revert|check <code dir> [<k1_transparency.py>]
Hooks are INERT until PROMOTION-GATE.json heldout.baseline == "prior-v1" (prior_gate.wanted / is_prior)."""
import os, sys

GS = [
    ("import heldout as HO\n",
     "import heldout as HO\nimport prior_gate as PG   # prior-v1 baseline (RA/20261101-baseline-switch): inert unless heldout.baseline == 'prior-v1'\n"),
    ("ROUNDS = None   # heldout.Rounds, created lazily (needs wj/rj/log/cr below)\n",
     "ROUNDS = None   # heldout.Rounds, created lazily (needs wj/rj/log/cr below)\ndef SVC(): return sys.modules[__name__]   # prior_gate calls back into this module\n"),
    ("def ensure_heldout_sample(e, H, f, d, sp):\n",
     "def ensure_heldout_sample(e, H, f, d, sp):\n    _pg = PG.hook_sample(SVC(), e, H, f, d, sp)   # prior-v1 (PASS = old path)\n    if _pg is not PG.PASS: return _pg\n"),
    ("    d = EP / str(e); dp = d / 'DECISION.json'; c = HO.cfg(f); R = rounds(); rid = s['round_id']; r = R.get(rid)\n",
     "    d = EP / str(e); dp = d / 'DECISION.json'; c = HO.cfg(f); R = rounds(); rid = s['round_id']; r = R.get(rid)\n"
     "    if PG.is_prior(r): return PG.process_epoch(SVC(), e, s, f)   # prior-v1 round\n"),
    ("        r = R.get(rid); closed = rj(R.d(rid) / 'CLOSED.json'); b = R.baseline(rid); turns = R.turns(rid)\n",
     "        r = R.get(rid)\n        if PG.is_prior(r): rows.append(PG.publish_round(SVC(), R, rid, r, out)); continue   # prior-v1 round\n"
     "        closed = rj(R.d(rid) / 'CLOSED.json'); b = R.baseline(rid); turns = R.turns(rid)\n"),
]
TP = [
    ("    if h.get('enabled') is not True: return None\n    try: cur = rj(HOME / 'subnet120/affine/website/gate/heldout/current.json')   # absent until the first round opens\n    except Exception: cur = None\n",
     "    if h.get('enabled') is not True: return None\n    try: cur = rj(HOME / 'subnet120/affine/website/gate/heldout/current.json')   # absent until the first round opens\n    except Exception: cur = None\n"
     "    if h.get('baseline') == 'prior-v1':   # prior-v1 baseline (RA/20261101-baseline-switch)\n"
     "        import prior_gate as _PG; return dict(_PG.public_params(h, cur), bar_z=h.get('bar_z', 1.0), z_inclusive=h.get('z_inclusive', False))\n"),
]
GL = [
    ("def K1_RULE_SUFFIX():\n",
     "def _prior_rule():\n    # prior-v1 baseline (RA/20261101-baseline-switch): rule text when PROMOTION-GATE.json heldout.baseline == 'prior-v1'\n"
     "    try:\n        f = json.loads(Path('/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json').read_text())\n"
     "        if (f.get('heldout') or {}).get('baseline') != 'prior-v1': return None\n        import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))\n"
     "        import prior_gate as _PG; return 'Promotion gate (held-out, ' + _PG.RULE + ') Earlier rule: ' + HELDOUT_RULE\n"
     "    except Exception: return None\n\n\ndef K1_RULE_SUFFIX():\n"),
    ("               rule=HELDOUT_RULE + ' History:", "               rule=(_prior_rule() or HELDOUT_RULE) + ' History:"),
]
K1 = [
    ("    if not (s and c and h and cp): return False\n",
     "    if not (s and c and h and cp): return False\n    if s.get('prior_v1'): return False   # prior-v1 rounds: reveal at /gate/heldout/<round>.json (prior_gate.publish_round)\n"),
    ("    rid = s.get('round_id')\n    if not (s.get('heldout') and rid",
     "    rid = s.get('round_id')\n    if s.get('prior_v1'): return False   # prior-v1 rounds: /gate/heldout/<round>.json carries tasks + GREEN grades\n    if not (s.get('heldout') and rid"),
]


def run(mode, path, pairs):
    s = open(path).read(); out = s
    for a, b in pairs:
        if mode in ('apply', 'check'):
            if out.count(b) == 1 and (a not in b or out.count(a) == 1): continue      # already applied
            if out.count(a) != 1: raise SystemExit('%s: anchor count %d for %r' % (path, out.count(a), a[:70]))
            out = out.replace(a, b)
        else:
            if out.count(b) == 0: continue
            if out.count(b) != 1: raise SystemExit('%s: patched block count %d' % (path, out.count(b)))
            out = out.replace(b, a)
    if mode == 'check': print('check ok %s' % path); return
    if out != s:
        t = path + '.tmp.pg'; open(t, 'w').write(out); os.chmod(t, os.stat(path).st_mode & 0o777); os.replace(t, path)
    print('%s %s%s' % (mode, path, '' if out != s else ' (no change)'))


if __name__ == '__main__':
    mode, code = sys.argv[1], sys.argv[2]
    run(mode, os.path.join(code, 'gate_service.py'), GS)
    run(mode, os.path.join(code, 'training_params.py'), TP)
    run(mode, os.path.join(code, 'gate_ledger.py'), GL)
    if len(sys.argv) > 3: run(mode, sys.argv[3], K1)
