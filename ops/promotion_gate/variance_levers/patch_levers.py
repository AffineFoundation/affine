#!/usr/bin/env python3
"""SN120 variance levers (Jacob, 2026-10-31 14:07 CR: "Go live with pooled sigma_run", "Go live with 8 candidate rollouts per task",
"Go live with 2,000 tasks per round"). Anchored, idempotent code patches; `revert` applies the exact reverse.
usage: patch_levers.py apply|revert FILE...   (files are matched by basename: heldout.py, gate_service.py)
Round-locking: k_cand and the sigma_run policy are read from the round record (ROUND.private.json), so a flag change only
affects rounds opened after it; rounds without a sigma_run_policy keep the old per-round rule max(floor, run_sigma)."""
import os, sys

HO_DEFAULTS_OLD = "                min_kept=100, scoring='ivw-v2', sigma_run=0.022)\n"
HO_DEFAULTS_NEW = ("                min_kept=100, scoring='ivw-v2', sigma_run=0.022,\n"
                   "                sigma_run_policy='per-round', sigma_pool_rounds=20, sigma_pool_min_rounds=5)\n"
                   "# variance levers (Jacob 2026-10-31 14:07 CR): sigma_run_policy 'pooled' = sigma_run of a round is the pooled run-level sd of the\n"
                   "# last sigma_pool_rounds finished baselines (pooled_sigma_run), frozen in ROUND.private.json when the round opens (max with the\n"
                   "# floor sigma_run); 'per-round' (old) = max(floor, this round's own 2-df estimate run_sigma). k_cand is also read from the round.\n")

HO_FUNCS_ANCHOR = "def settings_hash(settings: dict) -> str:\n"
HO_FUNCS_NEW = '''SIGMA_POOLED = 'pooled-v1'


def round_excess_var(sel: dict, blocks: dict, kept):
    """One round's run-level evidence from its p_i blocks: between-block sample variance of the block weighted means minus the
    mean within-block sampling variance (NOT floored), the within-block variance and the degrees of freedom (blocks - 1)."""
    ms, vs = [], []
    for pt in blocks.values():
        num = W = v = 0.0
        for t in kept:
            sc = sel.get(t) if t in sel else sel.get(str(t)); c = pt.get(t) if t in pt else pt.get(str(t))
            if not (isinstance(sc, dict) and isinstance(c, dict) and c.get('n')): continue
            w = weight(sc); x = c['c'] / c['n']; num += w * x; W += w; v += w * w * x * (1 - x) / max(1, c['n'] - 1)
        if W: ms.append(num / W); vs.append(v / (W * W))
    if len(ms) < 2: return None
    m = sum(ms) / len(ms); sv = sum((x - m) ** 2 for x in ms) / (len(ms) - 1); wv = sum(vs) / len(vs)
    return dict(excess=sv - wv, within=wv, df=len(ms) - 1)


def pooled_sigma_run(root, rj, max_rounds=20, exclude=()):
    """Pooled run-level sd over the most recent `max_rounds` rounds with a finished baseline (all p_i blocks present):
    excess = df-weighted mean of round_excess_var; sigma = sqrt(max(0, excess)); ub95 = one-sided 95 % upper bound
    (chi-square, Wilson-Hilferty). Rounds are read from <root>/r*/ (BASELINE.private.json selection + kept, BLOCK-1..). -> dict."""
    import math
    rows = []
    for p in sorted(Path(root).glob('r2*'), key=lambda q: q.name, reverse=True):
        if p.name in exclude: continue
        b = rj(p / 'BASELINE.private.json'); r = rj(p / 'ROUND.private.json') or {}
        if not (isinstance(b, dict) and b.get('per_task') and b.get('kept')): continue
        nb = int(r.get('blocks') or 0); blocks = {}
        for bi in range(1, nb):
            blk = rj(p / ('BLOCK-%d.private.json' % bi))
            if isinstance(blk, dict) and blk.get('per_task'): blocks[bi] = blk['per_task']
        if nb < 3 or len(blocks) != nb - 1: continue
        ev = round_excess_var(b.get('selection') or {}, blocks, b['kept'])
        if ev is None: continue
        rows.append(dict(ev, round=p.name))
        if len(rows) >= int(max_rounds): break
    df = sum(x['df'] for x in rows)
    if not df: return dict(rounds=0, df=0, round_ids=[], excess_var=None, within_var=None, sigma_run_est=None, ub95=None)
    ex = sum(x['df'] * x['excess'] for x in rows) / df; wv = sum(x['df'] * x['within'] for x in rows) / df
    z = 1.645; chi_lo = df * (1 - 2 / (9 * df) - z * math.sqrt(2 / (9 * df))) ** 3
    ub = math.sqrt(max(0.0, (ex + wv) * df / chi_lo - wv)) if chi_lo > 0 else None
    return dict(rounds=len(rows), df=df, round_ids=[x['round'] for x in rows], excess_var=ex, within_var=wv,
                sigma_run_est=math.sqrt(max(0.0, ex)), ub95=ub, positive_rounds=sum(x['excess'] > 0 for x in rows))


def sigma_policy(root, rj, c: dict) -> dict:
    """Frozen sigma_run policy of a round being opened (stored in ROUND.private.json). value None -> old per-round rule."""
    floor = float(c.get('sigma_run') or 0.0)
    try:
        pz = pooled_sigma_run(root, rj, int(c.get('sigma_pool_rounds') or 20))
    except Exception as ex:
        pz = dict(rounds=0, error=str(ex)[:200])
    ok = pz.get('sigma_run_est') is not None and int(pz.get('rounds') or 0) >= int(c.get('sigma_pool_min_rounds') or 5)
    return dict(pz, policy=SIGMA_POOLED, floor=floor, value=(max(floor, float(pz['sigma_run_est'])) if ok else None),
                fallback=(None if ok else 'per-round max(floor, run_sigma): too few pooled rounds'), computed_at=time.time(),
                rule='sigma_run = max(floor, sqrt(max(0, df-weighted mean over the last %d finished rounds of (between-block variance of '
                     'the 3 head p_i block weighted means - mean within-block sampling variance)))), frozen when the round opens'
                     % int(c.get('sigma_pool_rounds') or 20))


def sigma_policy_name(r: dict) -> str:
    p = r.get('sigma_run_policy') if isinstance(r, dict) else None
    return p['policy'] if isinstance(p, dict) and p.get('value') is not None else 'per-round'


def turn_sigma_run(r: dict, c: dict, b: dict) -> float:
    """sigma_run of a turn: the round's frozen pooled value, else the old per-round rule max(floor, run_sigma estimate)."""
    p = r.get('sigma_run_policy') if isinstance(r, dict) else None
    if isinstance(p, dict) and p.get('value') is not None: return float(p['value'])
    return max(float(c['sigma_run']), float((b or {}).get('sigma_run_est') or 0.0))


def public_sigma(r: dict):
    p = r.get('sigma_run_policy') if isinstance(r, dict) else None
    if not isinstance(p, dict): return dict(policy='per-round', rule='max(floor, this round\\'s 3-block estimate heldout.run_sigma)')
    return {k: p.get(k) for k in ('policy', 'value', 'floor', 'rounds', 'df', 'sigma_run_est', 'ub95', 'round_ids', 'fallback', 'rule')}


'''
HO_OPEN_OLD = "                 baseline_plan_seeds=[baseline_plan_seed(seed, b) for b in range(int(c['blocks']))])\n"
HO_OPEN_NEW = HO_OPEN_OLD + "        if c.get('sigma_run_policy') == 'pooled': r['sigma_run_policy'] = sigma_policy(self.root, self.rj, c)   # variance levers\n"
HO_PUB_OLD = "                    k_cand=r['k_cand'], bar_z=r['bar_z'], z_inclusive=r['z_inclusive'],\n"
HO_PUB_NEW = "                    k_cand=r['k_cand'], bar_z=r['bar_z'], z_inclusive=r['z_inclusive'], sigma_run=public_sigma(r),\n"
HO_RULE_OLD = "                         'GREEN iff gain > 0 and z > bar_z and estimated cap-hit < 0.20.')\n"
HO_RULE_NEW = ("                         'GREEN iff gain > 0 and z > bar_z and estimated cap-hit < 0.20; z = gain / sqrt(se_task^2 + sigma_run^2 (1 + 1/3)), '\n"
               "                         'sigma_run per the round\\'s sigma_run field (pooled over recent rounds, frozen at open).')\n")

GS_K_OLD = "    kp = dict(k=int(c['k_cand']), seeded=True); pseed = HO.cand_plan_seed(s['seed'])\n"
GS_K_NEW = "    kp = dict(k=int(r.get('k_cand') or c['k_cand']), seeded=True); pseed = HO.cand_plan_seed(s['seed'])   # k_cand locked per round\n"
GS_S_OLD = "sigma_run=max(float(c['sigma_run']), float(bfull.get('sigma_run_est') or 0.0)), n_p_blocks=int(r['blocks']) - 1)\n"
GS_S_NEW = "sigma_run=HO.turn_sigma_run(r, c, bfull), n_p_blocks=int(r['blocks']) - 1)\n    dec['evaluated']['sigma_run_policy'] = HO.sigma_policy_name(r)\n"
GS_H_OLD = "                            scoring='ivw-v2', settings_hash=b.get('settings_hash')))\n"
GS_H_NEW = "                            scoring='ivw-v2', settings_hash=b.get('settings_hash'), sigma_run_policy=HO.sigma_policy_name(r)))\n"
GS_R_OLD = "head and candidate scored on the same tasks of a private held-out probe set (4 seeded samples per task, shared seeds);"
GS_R_NEW = ("head and candidate scored on the same tasks of a private held-out probe set (seeded samples per task, shared seeds; set size n, "
            "candidate k_cand and the sigma_run policy are fixed per round and published in /gate/heldout/current.json: 1000 tasks, k_cand 4, "
            "per-round sigma_run until the variance-levers switch of 2026-10-31, then 2000 tasks, k_cand 8, pooled sigma_run);")
GS_DOC_OLD = "  candidate of the round probed on S minus its batch (k=4, fresh plan seed) and decided vs that baseline with z >= bar_z.\n"
GS_DOC_NEW = GS_DOC_OLD + "  Variance levers (2026-10-31): n, k_cand and the sigma_run policy are frozen per round in ROUND.private.json (heldout.py).\n"

PAIRS = {'heldout.py': [(HO_DEFAULTS_OLD, HO_DEFAULTS_NEW), (HO_FUNCS_ANCHOR, HO_FUNCS_NEW + HO_FUNCS_ANCHOR), (HO_OPEN_OLD, HO_OPEN_NEW),
                        (HO_PUB_OLD, HO_PUB_NEW), (HO_RULE_OLD, HO_RULE_NEW)],
         'gate_service.py': [(GS_K_OLD, GS_K_NEW), (GS_S_OLD, GS_S_NEW), (GS_H_OLD, GS_H_NEW), (GS_R_OLD, GS_R_NEW), (GS_DOC_OLD, GS_DOC_NEW)]}

if __name__ == '__main__':
    mode = sys.argv[1]; rc = 0
    for p in sys.argv[2:]:
        s = open(p).read(); o = s
        for old, new in PAIRS[os.path.basename(p)]:
            a, b = (old, new) if mode == 'apply' else (new, old)
            if b in s and (a not in s or a in b):
                continue      # already in target state
            if s.count(a) != 1:
                print('ANCHOR %s: %r found %d times' % (p, a[:70], s.count(a))); rc = 2; continue
            s = s.replace(a, b)
        if rc: print('NOT WRITTEN %s' % p); continue
        if s != o:
            t = p + '.tmp-vl'; open(t, 'w').write(s); os.chmod(t, os.stat(p).st_mode & 0o777)
            import py_compile; py_compile.compile(t, doraise=True, cfile=t + 'c'); os.unlink(t + 'c')
            os.replace(t, p); print('%s %s' % (mode, p))
        else:
            print('unchanged %s' % p)
    sys.exit(rc)
