#!/usr/bin/env python3
"""Per-miner aggregation of the max-group rule (CPU only, stored gradient-dots).
Group score s_g = w . (sum_i A_i d_i) / (8192 n)  (same reference weighting as the pipeline). Per epoch, honest real groups are split
in two halves: CAL (calibrates the null) and SIM (background for synthetic miners), so calibration is out-of-sample.
Batch statistic: M = max_g s_g over its S groups; batch p-value p = P_null(M >= m) from 200k CAL batches of the same size S.
Batch 'hit' = p < q where q = FP rate of the old 'z>2' max-group threshold at that size.
Miner tests over its B batches (alpha = 0.01 per miner):
  any      : flagged if any batch hits (naive, what a per-batch rule does)
  binom    : P(Binom(B, q) >= hits) < alpha
  stouffer : sum Phi^-1(1-p_b) / sqrt(B) > Phi^-1(1-alpha)
  naive_sumz: sum of raw max-group z_b / sqrt(B) > 2.33 (z_b against median/sd of the null, as in 'Detection threshold')
Synthetic miners: B batches (B=3..12) of S=3 or 9 groups, k=0 (honest), 1 or 2 beyond-policy (bo16) groups per batch.
Gaming: an honest miner with a fixed budget of 36 groups split into batches of 1, 2, 3, 4, 6, 9, 12, 18, 36 groups.
Real check: actual UIDs, one batch per epoch over e62..e66, aggregated with the same tests."""
import glob, gzip, json, sys, collections
from pathlib import Path
import numpy as np
from scipy.stats import norm, binom
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import grader, grad_real
from plant_dots import key
rew = lambda c, L: grader.overlong_reward(c, L)
rng = np.random.default_rng(20261019); ALPHA = 0.01; NNULL = 200000; NMIN = 2000
def load(E, planted=True):
    D = [json.load(open(f)) for f in sorted(glob.glob(str(HERE / 'data' / f'e{E}' / 'dots-real-*.json')))]; ch = D[0]['chunks']
    sub = [k for k, c in enumerate(ch) if c['kind'] == 'subject']; h0 = [k for k in sub if ch[k]['half'] == 0]; h1 = [k for k in sub if ch[k]['half'] == 1]
    w = np.zeros(len(ch)); w[h0] = 0.5 / len(h0); w[h1] = 0.5 / len(h1)
    def gs(rws, dots):
        r = np.asarray(rws); return float(w @ ((r - r.mean()) @ np.asarray(dots))) / (8192 * len(r))
    hon, byu = [], collections.defaultdict(list)
    for d in D:
        for g in d['groups']:
            s = gs([rew(1.0 if c == 'positive' else 0.0, L) for c, L in zip(g['claimed_cls'], g['lens'])], g['dots']); hon.append(s)
            if g['uid'] is not None: byu[g['uid']].append(s)
    bp = []
    P = HERE / 'data' / 'planted' / f'e{E}'
    if planted and (P / 'teacher.json').exists():
        base, meta = grad_real.load_ref(E); cache = {}
        for f in glob.glob(str(P / 'cache-*.json')): cache.update(json.load(open(f)))
        teach = json.load(open(P / 'teacher.json')); tz = np.load(P / 'teacher.npz'); to = tz['offsets']; tt = tz['tokens']
        byref = collections.defaultdict(list)
        for x in meta:
            if x['kind'] == 'ref': byref[x['index']].append(x)
        cor = collections.defaultdict(list)
        for k, c in enumerate(teach['correct']): cor[c['index']].append((c['j'], tt[to[k]:to[k + 1]]))
        for i, cs in cor.items():
            cs = sorted([c for c in cs if c[0] < 16], key=lambda c: c[0])[:2]
            if not cs: continue
            wr = [x for x in byref[i] if x['outcome'] != 'positive'][:6]
            toks = [c[1] for c in cs] + [x['tokens'] for x in wr]; rws = [rew(1.0, len(c[1])) for c in cs] + [rew(0.0, x['len']) for x in wr]
            dts = [cache.get(key(i, t)) for t in toks]
            if any(d is None for d in dts): continue
            bp.append(gs(rws, dts))
    return np.array(hon), {u: np.array(v) for u, v in byu.items()}, np.array(bp)
class Null:
    def __init__(self, cal):
        self.cal = cal; self.t = {}
    def get(self, S):
        if S not in self.t:
            m = np.sort(self.cal[rng.integers(0, len(self.cal), (NNULL, S))].max(1)); med, sd = float(np.median(m)), float(m.std())
            q = float((m > med + 2 * sd).mean()); self.t[S] = (m, med, sd, q)
        return self.t[S]
    def p(self, S, M):
        m = self.t[S][0] if S in self.t else self.get(S)[0]
        return np.clip(1.0 - np.searchsorted(m, M, side='left') / len(m), 0.5 / len(m), 1.0)
def tests(null, S, M):   # M: [miners, B] batch maxima
    m, med, sd, q = null.get(S); p = null.p(S, M); B = M.shape[1]; hits = (p < q).sum(1)
    return dict(any=float((hits > 0).mean()), binom=float((binom.sf(hits - 1, B, q) < ALPHA).mean()),
                stouffer=float((norm.isf(p).sum(1) / np.sqrt(B) > norm.isf(ALPHA)).mean()), naive_sumz=float((((M - med) / sd).sum(1) / np.sqrt(B) > norm.isf(ALPHA)).mean()))
def miners(sim, bp, S, B, k, n=NMIN):
    g = sim[rng.integers(0, len(sim), (n, B, S))]
    if k: g[:, :, :k] = bp[rng.integers(0, len(bp), (n, B, k))]
    return g.max(2)
out = dict(alpha=ALPHA, epochs={}, gaming={}, real={})
REAL = {}
for E in ['62', '63', '64', '65', '66']:
    if not glob.glob(str(HERE / 'data' / f'e{E}' / 'dots-real-*.json')): continue
    hon, byu, bp = load(E, planted=E in ('62', '64', '66')); perm = rng.permutation(len(hon)); cal, sim = hon[perm[::2]], hon[perm[1::2]]
    REAL[E] = (Null(hon), byu)
    if not len(bp): continue
    null = Null(cal); res = dict(n_groups=len(hon), bp_pool=len(bp), q={S: null.get(S)[3] for S in (1, 2, 3, 4, 6, 9, 12, 18, 36)}, grid={})
    for S in (3, 9):
        for k in (0, 1, 2):
            for B in (1, 3, 4, 6, 8, 10, 12):
                res['grid'][f'S{S}_k{k}_B{B}'] = tests(null, S, miners(sim, bp, S, B, k))
    # gaming: honest miner, fixed 36-group budget, split into 36/S batches of S groups
    gm = {}
    for S in (1, 2, 3, 4, 6, 9, 12, 18, 36):
        gm[S] = tests(null, S, miners(sim, bp, S, 36 // S, 0)); gm[S]['batches'] = 36 // S; gm[S]['exp_hits'] = (36 // S) * null.get(S)[3]
    # budget-matched power: 36-group budget with 2 beyond-policy groups per 9 groups (8 total), split differently
    pw = {}
    for S in (3, 9):
        nb = 36 // S; kk = 2 if S == 9 else 1   # S=3: 1 bp per 3-group batch = 12 bp total; S=9: 2 bp per batch = 8 total
        pw[f'S{S}_k{kk}'] = tests(null, S, miners(sim, bp, S, nb, kk))
    res['budget36_power'] = pw; out['epochs'][E] = res; out['gaming'][E] = gm
    print('== E', E, 'groups', len(hon), 'bp', len(bp), 'q', {S: round(v, 3) for S, v in res['q'].items()}, flush=True)
    for S in (3, 9):
        for k in (0, 1, 2):
            print(f' S{S} k{k}', ' '.join(f"B{B}:" + '/'.join(f"{res['grid'][f'S{S}_k{k}_B{B}'][t]:.2f}" for t in ('any', 'binom', 'stouffer', 'naive_sumz')) for B in (1, 3, 4, 6, 8, 10, 12)))
    print(' gaming (any/binom/stouffer/naive_sumz)', {S: '/'.join(f"{v[t]:.3f}" for t in ('any', 'binom', 'stouffer', 'naive_sumz')) for S, v in gm.items()})
    print(' budget36', {k: '/'.join(f"{v[t]:.2f}" for t in ('any', 'binom', 'stouffer')) for k, v in pw.items()})
# real UIDs: one batch per epoch, null = that epoch's honest groups at the UID's batch size
uids = sorted(set().union(*[set(b) for _, b in REAL.values()]), key=str); rows = []
for u in uids:
    ps, zs, hits, qs = [], [], 0, []
    for E, (nl, byu) in REAL.items():
        if u not in byu: continue
        S = len(byu[u]); m, med, sd, q = nl.get(S); M = byu[u].max(); p = float(nl.p(S, np.array([M]))[0])
        ps.append(p); zs.append((M - med) / sd); hits += p < q; qs.append(q)
    if len(ps) < 3: continue
    B = len(ps); qm = float(np.mean(qs))
    rows.append(dict(B=B, hits=int(hits), any=hits > 0, binom=bool(binom.sf(hits - 1, B, qm) < ALPHA), stouffer=bool(norm.isf(np.array(ps)).sum() / np.sqrt(B) > norm.isf(ALPHA)),
                     naive_sumz=bool(np.sum(zs) / np.sqrt(B) > norm.isf(ALPHA)), stouffer_z=float(norm.isf(np.array(ps)).sum() / np.sqrt(B))))
out['real'] = dict(n_uids=len(rows), mean_batches=float(np.mean([r['B'] for r in rows])), flag_rate={t: float(np.mean([r[t] for r in rows])) for t in ('any', 'binom', 'stouffer', 'naive_sumz')},
                   hits_hist=dict(collections.Counter(r['hits'] for r in rows)), stouffer_z_sd=float(np.std([r['stouffer_z'] for r in rows])), stouffer_z_mean=float(np.mean([r['stouffer_z'] for r in rows])))
print('== real UIDs (one batch per epoch, e62..e66)', json.dumps(out['real']))
json.dump(out, open(HERE / 'data' / 'miner_agg.json', 'w'), default=str)
