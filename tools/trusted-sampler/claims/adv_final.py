#!/usr/bin/env python3
"""Consolidated v9 threat table (CPU, stored dots). 5 random CAL/SIM splits of each epoch's honest real groups; 3-group batches, B=10 per miner;
gate = Stouffer on max-group p AND binomial hit test, both at alpha=0.01/210 (v9 as specified). Reports gated rate [min,max] and mean payment
V = sum_g (z_g - median) clipped at 0 for gated miners (units: honest group SD; one beyond-policy k=1 miner ~ reference).
Leakage: 'live seed' = attacker reproduces the exact reference (H('ref',E)/H('chunks',E) are public) -> selects on the exact score (full->full).
'post-commit draw' = attacker only knows the env distribution -> builds its own reference from an independent same-distribution draw
(half 0) and the validator scores on an independent draw (half 1); matched control 'exact h1->h1' uses the same half-size reference."""
import glob, gzip, json, sys, collections
from pathlib import Path
import numpy as np
from scipy.stats import norm, binom
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import grader, grad_real
from plant_dots import key
rew = lambda c, L: grader.overlong_reward(c, L)
MODE = sys.argv[1] if len(sys.argv) > 1 else 'none'
rng = np.random.default_rng(2026); NM = 2000; ABON = 0.01 / 210; NSPLIT = 5; B = 10
T = json.load(open(HERE.parent / 'data/original-math7496.tasks.json'))
class Null:
    def __init__(self, cal, n=200000):
        self.m = np.sort(cal[rng.integers(0, len(cal), (n, 3))].max(1)); self.q = float((self.m > np.median(self.m) + 2 * self.m.std()).mean())
    def p(self, M): return np.clip(1.0 - np.searchsorted(self.m, M, side='left') / len(self.m), 0.5 / len(self.m), 1 - 0.5 / len(self.m))
def gate(nl, Sc):
    p = nl.p(Sc.max(2)); z = norm.isf(p).sum(1) / np.sqrt(B); h = (p < nl.q).sum(1)
    g = (z > norm.isf(ABON)) & (binom.sf(h - 1, B, nl.q) < ABON); V = np.where(g, np.clip(Sc.sum((1, 2)), 0, None), 0.0)
    return float(g.mean()), float(V.mean()), float((z > norm.isf(0.01)).mean())
def gnk(ks, r, SK):
    sk = [SK.get(k) for k in ks]
    return float(np.linalg.norm((r - r.mean()) @ np.stack(sk))) / (8192 * len(r)) if all(v is not None for v in sk) else np.nan
def extra_groups(E, name, cache, byref, sel=None, SK={}):
    A = HERE / 'data' / 'adv' / f'e{E}'
    if not (A / f'{name}.json').exists(): return []
    dd = {}
    for f in glob.glob(str(A / f'dots-{name}-*.json')): dd.update(json.load(open(f)))
    m = json.load(open(A / f'{name}.json'))['meta']; z = np.load(A / f'{name}.npz'); o = z['offsets']; tk = z['tokens']; out = []
    if name.startswith('replay'):
        by = collections.defaultdict(list)
        for n, x in enumerate(m): by[x['index']].append((x, key(x['index'], tk[o[n]:o[n + 1]])))
        for i, xs in by.items():
            if all(k in dd for _, k in xs):
                r = np.array([x['reward'] for x, _ in xs]); out.append((((r - r.mean()) @ np.array([dd[k] for _, k in xs])) / (8192 * len(r)), gnk([k for _, k in xs], r, SK)))
        return out
    by = collections.defaultdict(list)
    for n, x in enumerate(m):
        k = key(x['index'], tk[o[n]:o[n + 1]])
        if x['outcome'] == 'positive' and k in dd and (sel is None or x.get('variant') == sel): by[x['index']].append((x, k))
    for i, xs in by.items():
        wr = [x for x in byref.get(i, []) if x['outcome'] != 'positive'][:6]
        wk = [cache.get(key(i, x['tokens'])) for x in wr]
        if len(wr) < 6 or any(v is None for v in wk): continue
        c2 = xs[:2]; r = np.array([rew(1.0, x['len']) for x, _ in c2] + [rew(0.0, x['len']) for x in wr])
        out.append((((r - r.mean()) @ np.array([dd[k] for _, k in c2] + wk)) / (8192 * len(r)), gnk([k for _, k in c2] + [key(i, x['tokens']) for x in wr], r, SK)))
    return out
OUT = {}
for E in ('62', '64', '66'):
    D = [json.load(open(f)) for f in sorted(glob.glob(str(HERE / 'data' / f'e{E}' / 'dots-real-*.json')))]; ch = D[0]['chunks']
    sub = [k for k, c in enumerate(ch) if c['kind'] == 'subject']; h0 = [k for k in sub if ch[k]['half'] == 0]; h1 = [k for k in sub if ch[k]['half'] == 1]
    reft = {i for c in ch for i in c['tasks']}
    R = {(r['miner'], r['slot']): r for r in (json.loads(l) for l in gzip.open(HERE / 'data/real' / f'real-{E}.jsonl.gz', 'rt'))}
    SK = {}
    for f in glob.glob(str(HERE / 'data' / 'gnorm' / f'e{E}' / 'sk-*.npz')):
        z = np.load(f); SK.update(zip(z['keys'].tolist(), z['sk']))
    VEC, gn, ln, idx, pas = [], [], [], [], []
    for d in D:
        for g in d['groups']:
            r = R[(g['miner'], g['slot'])]; rws = np.array([rew(1.0 if c == 'positive' else 0.0, L) for c, L in zip(g['claimed_cls'], g['lens'])]); A = rws - rws.mean()
            VEC.append((A @ np.array(g['dots'])) / (8192 * len(rws))); ln.append(np.mean(g['lens'])); idx.append(r['index']); pas.append(np.mean([c == 'positive' for c in g['claimed_cls']]))
            sks = [SK.get(key(r['index'], np.asarray(x['out'], dtype=np.int32))) for x in r['rollouts']]
            gn.append(float(np.linalg.norm(A @ np.stack(sks))) / (8192 * len(rws)) if all(v is not None for v in sks) else np.nan)
    VEC, gn, ln, idx, pas = map(np.array, (VEC, gn, ln, idx, pas)); subj = np.array([T[i]['data']['subject'] for i in idx]); NR = len(VEC)
    P = HERE / 'data' / 'planted' / f'e{E}'; cache = {}
    for f in glob.glob(str(P / 'cache-*.json')): cache.update(json.load(open(f)))
    teach = json.load(open(P / 'teacher.json')); tz = np.load(P / 'teacher.npz'); to = tz['offsets']; tt = tz['tokens']; base, meta = grad_real.load_ref(E)
    byref = collections.defaultdict(list)
    for x in meta:
        if x['kind'] == 'ref': byref[x['index']].append(x)
    cor = collections.defaultdict(list)
    for kk, c in enumerate(teach['correct']):
        if c['j'] < 16: cor[c['index']].append((c['j'], tt[to[kk]:to[kk + 1]]))
    EXTRA = {'beyond_policy_bo16': []}
    for i, cs in cor.items():
        cs = sorted(cs, key=lambda c: c[0])[:2]; wr = [x for x in byref[i] if x['outcome'] != 'positive'][:6]
        toks = [c[1] for c in cs] + [x['tokens'] for x in wr]; r = np.array([rew(1.0, len(c[1])) for c in cs] + [rew(0.0, x['len']) for x in wr])
        dts = [cache.get(key(i, t)) for t in toks]
        if all(v is not None for v in dts): EXTRA['beyond_policy_bo16'].append((((r - r.mean()) @ np.array(dts)) / (8192 * len(r)), gnk([key(i, t) for t in toks], r, SK)))
    for nm, sel in (('copies', 'hint'), ('copies', 'prefix50'), ('distill', None), ('replay62', None)):
        g = extra_groups(E, nm, cache, byref, sel, SK)
        if g: EXTRA[f'{nm}_{sel}' if sel else nm] = g
    keep = ~np.isnan(gn) if MODE != 'none' else np.ones(NR, bool)
    VEC, gn, ln, idx, pas, subj = VEC[keep], gn[keep], ln[keep], idx[keep], pas[keep], subj[keep]; NR = len(VEC)
    off = {}; ALLV = [VEC]; GN = [gn]; n0 = NR
    for k, v in EXTRA.items():
        v = [x for x in v if MODE == 'none' or not np.isnan(x[1])]
        if not v: continue
        off[k] = (n0, len(v)); ALLV.append(np.array([x[0] for x in v])); GN.append(np.array([x[1] for x in v])); n0 += len(v)
    ALLV = np.concatenate(ALLV); GN = np.concatenate(GN)
    if MODE == 'cos': ALLV = ALLV / GN[:, None]
    W = {'full': np.zeros(len(ch)), 'h0': np.zeros(len(ch)), 'h1': np.zeros(len(ch))}
    W['full'][h0] = 0.5 / len(h0); W['full'][h1] = 0.5 / len(h1); W['h0'][h0] = 1 / len(h0); W['h1'][h1] = 1 / len(h1)
    RES = collections.defaultdict(list)
    for rep in range(NSPLIT):
        perm = rng.permutation(NR); cal, sim = perm[::2], perm[1::2]; sim = sim[~np.isin(idx[sim], list(reft))]
        S, NL = {}, {}
        for k, w in W.items():
            s = ALLV @ w
            if MODE == 'cond':   # calibrate each group against CAL groups in the same gradient-norm decile
                ed = np.quantile(GN[cal], np.linspace(0, 1, 11)[1:-1]); bn = np.searchsorted(ed, GN); s2 = np.empty_like(s)
                for b_ in range(10):
                    cb = cal[bn[cal] == b_]; s2[bn == b_] = (s[bn == b_] - np.median(s[cb])) / s[cb].std()
                s = s2
            s = (s - np.median(s[cal])) / s[cal].std(); S[k] = s; NL[k] = Null(s[cal])
        def H(n=NM): return sim[rng.integers(0, len(sim), (n, B, 3))]
        sc = {'honest': ('full', H())}
        for k, (o_, n_) in off.items():
            for kk in ((1, 2) if k == 'beyond_policy_bo16' else (1,)):
                I = H(); I[:, :, :kk] = o_ + rng.integers(0, n_, (NM, B, kk)); sc[f'{k}_k{kk}'] = ('full', I)
        for n in (8, 32, 64):
            dr = sim[rng.integers(0, len(sim), (NM, B, 3 * n))]
            top = lambda pr: np.take_along_axis(dr, np.argsort(-pr[dr], axis=2)[:, :, :3], 2)
            sc[f'live_seed_exact_x{n}'] = ('full', top(S['full']))
            sc[f'postcommit_proxy_x{n}'] = ('h1', top(S['h0']))
            sc[f'matched_exact_h1_x{n}'] = ('h1', top(S['h1']))
            sc[f'length_refree_x{n}'] = ('full', top(ln))
            sc[f'gnorm_refree_x{n}'] = ('full', top(gn))
        sm = {s_: float(S['h0'][sim][subj[sim] == s_].mean()) for s_ in set(subj)}; best = max(sm, key=sm.get); tuned = sim[subj[sim] == best]
        sc['subject_guess_postcommit'] = ('h1', tuned[rng.integers(0, len(tuned), (NM, B, 3))])
        easy = sim[pas[sim] >= 0.75]; sc['easy_tasks'] = ('full', easy[rng.integers(0, len(easy), (NM, B, 3))])
        for nm, (k, I) in sc.items(): RES[nm].append(gate(NL[k], S[k][I]))
    OUT[E] = {nm: dict(gated=float(np.mean([a for a, _, _ in v])), gated_min=float(np.min([a for a, _, _ in v])), gated_max=float(np.max([a for a, _, _ in v])),
                       V=float(np.mean([b for _, b, _ in v])), stouffer01=float(np.mean([c for _, _, c in v]))) for nm, v in RES.items()}
    OUT[E]['_pools'] = {k: v[1] for k, v in off.items()}
    print('== E', E, OUT[E]['_pools'], flush=True)
    for nm, r in OUT[E].items():
        if nm[0] != '_': print(f"  {nm:28s} gated={r['gated']:.2f} [{r['gated_min']:.2f},{r['gated_max']:.2f}] V={r['V']:.1f} st01={r['stouffer01']:.2f}", flush=True)
json.dump(OUT, open(HERE / 'data' / ('adv_final.json' if MODE == 'none' else f'adv_final_{MODE}.json'), 'w'))
