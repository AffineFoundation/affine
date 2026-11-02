#!/usr/bin/env python3
"""Patch RA/20261027-k1-per-uid/code/k1_service.py: payee share = (1/N) x m_i renormalized (m from RA/20261102-gain-mult).
All m == 1 (today) -> the original code path (s = 1.0 / N), bit-identical shares. Idempotent (marker). usage: patch_k1_gainmult.py FILE"""
import sys
P = sys.argv[1]
s = open(P).read()
MARK = '# gain-mult 20261102'
if MARK in s:
    print('already patched'); sys.exit(0)

def rep(old, new):
    global s
    assert s.count(old) == 1, ('anchor count', s.count(old), old[:80])
    s = s.replace(old, new)

rep("import hashlib, json, os, sys, time, traceback\n",
    "import hashlib, json, math, os, sys, time, traceback\n")
rep("GRANTS = Path(",
    "GAINM = Path('/home/const/subnet120-rewrite/state/root-audits/20261102-gain-mult')   " + MARK + " (Jacob pt1136u/pt1137u)\n"
    "GRANTS = Path(")
rep("def scores_doc(recs, chain, q):\n",
    "def gain_mults():\n"
    "    \"\"\"" + MARK + ": -> {green_epoch: (m, settled)} for kings crowned after the activation epoch (RA/20261102-gain-mult). Missing,\n"
    "    disabled, malformed or out-of-range -> {} / m = 1 (fail-safe: the equal 1/N split). Kings at or before activation are never listed.\"\"\"\n"
    "    try:\n"
    "        cfg = rj(GAINM / 'code' / 'config.json') or {}\n"
    "        if cfg.get('enabled') is not True or cfg.get('activation_epoch') is None: return {}\n"
    "        a = int(cfg['activation_epoch']); lo, hi = float(cfg.get('m_min', 0.5)), float(cfg.get('m_max', 2.0))\n"
    "        doc = rj(GAINM / 'state' / 'GAIN-MULT.private.json') or {}\n"
    "        out = {}\n"
    "        for k in doc.get('kings') or []:\n"
    "            e, m = int(k['epoch']), float(k['m'])\n"
    "            if e <= a or not math.isfinite(m) or not (lo <= m <= hi): continue\n"
    "            out[e] = (m if k.get('settled') is True else 1.0, k.get('settled') is True)\n"
    "        return out\n"
    "    except Exception as x:\n"
    "        log(event='gain_mult_error', error=repr(x)[:300]); return {}\n\n\n"
    "def scores_doc(recs, chain, q):\n")
rep("""    if valid:
        s = 1.0 / len(valid)
        for r in valid:
            c = chain['uids'][str(r['uid'])]
            miners.append(dict(miner_hex=c['miner_hex'], uid=int(r['uid']), hotkey=r['hotkey'], share_now=s, flip_score=s,
                               k1_green_epoch=r['epoch_index'], k1_z=r.get('z'), banned=False, floor_share=False))
""", """    gm = gain_mults()   # """ + MARK + """
    if valid:
        ms = [gm.get(r['epoch_index'], (1.0, False))[0] for r in valid]
        if all(m == 1.0 for m in ms): shs = [1.0 / len(valid)] * len(valid)   # unchanged path (bit-identical 1/N)
        else: tot = math.fsum(ms); shs = [m / tot for m in ms]                # (1/N) x m_i renormalized: payee total unchanged
        for r, s, m in zip(valid, shs, ms):
            c = chain['uids'][str(r['uid'])]
            miners.append(dict(miner_hex=c['miner_hex'], uid=int(r['uid']), hotkey=r['hotkey'], share_now=s, flip_score=s,
                               k1_green_epoch=r['epoch_index'], k1_z=r.get('z'), banned=False, floor_share=False,
                               gain_m=round(m, 3), gain_settled=gm.get(r['epoch_index'], (1.0, None))[1]))
""")
rep("""                k1=dict(payees=[dict(uid=r['uid'], hotkey=r['hotkey'], green_epoch=r['epoch_index'], z=r.get('z'),
                                     still_registered=r in valid) for r in allp],""",
    """                k1=dict(payees=[dict(uid=r['uid'], hotkey=r['hotkey'], green_epoch=r['epoch_index'], z=r.get('z'),
                                     still_registered=r in valid, gain_m=round(gm.get(r['epoch_index'], (1.0, None))[0], 3),
                                     gain_settled=gm.get(r['epoch_index'], (1.0, None))[1]) for r in allp],""")
rep("""                                   'pre-K=1 scores are kept.'),""",
    """                                   'pre-K=1 scores are kept. ' + GAIN_TEXT),""")
rep("def gain_mults():\n",
    "GAIN_TEXT = ('King gain multiplier (Jacob 2026-11-02, " + MARK.split(' ')[1] + "): each payee crowned after the activation epoch gets weight (1/N) x m, '\n"
    "             'renormalized so the payees total is unchanged; m = clamp(1 + (cons / median - 1), 0.5, 2), cons = max(0, gain - 1 SE), gain = the '\n"
    "             'king pass rate minus its parent pass rate on the gate-pool tasks the prior sampler ran for both (same token cap); m = 1 until >= 2000 '\n"
    "             'shared tasks (gain_settled false); kings crowned at or before the activation epoch keep m = 1 (gain_settled null). '\n"
    "             'Per-king m: miners[].gain_m, k1.payees[].gain_m, https://affine.io/mining.json training.k1.king_gain_multiplier.')\n\n\n"
    "def gain_mults():\n")
rep("formula=dict(version='k1-last10-green-equal-v1',", "formula=dict(version='k1-last10-green-gainmult-v1',")
open(P, 'w').write(s)
print('patched')
