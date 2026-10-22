#!/usr/bin/env python3
"""(adv_cpu5) Miner-level self-studentized tests vs variance lottery, plus (adv_cpu4) REPLICATION gate vs variance-lottery attacks (CPU). Two independent secret references = the two disjoint task halves.
Each miner's batches are scored separately against half A and half B (each with its own out-of-sample max-of-3 null); the miner is gated only if
BOTH halves pass Stouffer+binomial at alpha_each = sqrt(0.01/210) (independent halves => family-wise ~0.01/210). Compared with the single
full-reference gate at 0.01/210. Selection is index-based so both halves see the same submitted groups. 3 CAL/SIM splits."""
import glob, gzip, json, sys, collections
from pathlib import Path
import numpy as np
from scipy.stats import norm, binom, t as tdist, wilcoxon
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import grader, grad_real
from plant_dots import key
rew = lambda c, L: grader.overlong_reward(c, L)
rng = np.random.default_rng(13); NM = 2000; ABON = 0.01 / 210; AE = np.sqrt(ABON)
class Null:
    def __init__(self, cal, n=200000):
        self.m = np.sort(cal[rng.integers(0, len(cal), (n, 3))].max(1)); self.q = float((self.m > np.median(self.m) + 2 * self.m.std()).mean())
    def p(self, M): return np.clip(1.0 - np.searchsorted(self.m, M, side='left') / len(self.m), 0.5 / len(self.m), 1 - 0.5 / len(self.m))
def gv(nl, Sc, a):
    p = nl.p(Sc.max(2)); B = Sc.shape[1]; z = norm.isf(p).sum(1) / np.sqrt(B); h = (p < nl.q).sum(1)
    return (z > norm.isf(a)) & (binom.sf(h - 1, B, nl.q) < a)
out = {}
for E in ('62', '64', '66'):
    D = [json.load(open(f)) for f in sorted(glob.glob(str(HERE / 'data' / f'e{E}' / 'dots-real-*.json')))]; ch = D[0]['chunks']
    sub = [k for k, c in enumerate(ch) if c['kind'] == 'subject']; h0 = [k for k in sub if ch[k]['half'] == 0]; h1 = [k for k in sub if ch[k]['half'] == 1]
    reft = {i for c in ch for i in c['tasks']}
    R = {(r['miner'], r['slot']): r for r in (json.loads(l) for l in gzip.open(HERE / 'data/real' / f'real-{E}.jsonl.gz', 'rt'))}
    SK = {}
    for f in glob.glob(str(HERE / 'data' / 'gnorm' / f'e{E}' / 'sk-*.npz')):
        z = np.load(f); SK.update(zip(z['keys'].tolist(), z['sk']))
    VEC, gn, ln, idx = [], [], [], []
    for d in D:
        for g in d['groups']:
            r = R[(g['miner'], g['slot'])]; rws = np.array([rew(1.0 if c == 'positive' else 0.0, L) for c, L in zip(g['claimed_cls'], g['lens'])]); A = rws - rws.mean()
            VEC.append((A @ np.array(g['dots'])) / (8192 * len(rws))); ln.append(np.mean(g['lens'])); idx.append(r['index'])
            sks = [SK.get(key(r['index'], np.asarray(x['out'], dtype=np.int32))) for x in r['rollouts']]
            gn.append(float(np.linalg.norm(A @ np.stack(sks))) if all(v is not None for v in sks) else -1.0)
    VEC, gn, ln, idx = map(np.array, (VEC, gn, ln, idx))
    P = HERE / 'data' / 'planted' / f'e{E}'; cache = {}
    for f in glob.glob(str(P / 'cache-*.json')): cache.update(json.load(open(f)))
    teach = json.load(open(P / 'teacher.json')); tz = np.load(P / 'teacher.npz'); to = tz['offsets']; tt = tz['tokens']; base, meta = grad_real.load_ref(E)
    byref = collections.defaultdict(list)
    for x in meta:
        if x['kind'] == 'ref': byref[x['index']].append(x)
    cor = collections.defaultdict(list)
    for kk, c in enumerate(teach['correct']):
        if c['j'] < 16: cor[c['index']].append((c['j'], tt[to[kk]:to[kk + 1]]))
    BPV = []
    for i, cs in cor.items():
        cs = sorted(cs, key=lambda c: c[0])[:2]; wr = [x for x in byref[i] if x['outcome'] != 'positive'][:6]
        toks = [c[1] for c in cs] + [x['tokens'] for x in wr]; rws = np.array([rew(1.0, len(c[1])) for c in cs] + [rew(0.0, x['len']) for x in wr])
        dts = [cache.get(key(i, t)) for t in toks]
        if all(v is not None for v in dts): BPV.append(((rws - rws.mean()) @ np.array(dts)) / (8192 * len(rws)))
    BPV = np.array(BPV); NB = len(BPV)
    ALLV = np.concatenate([VEC, BPV]); isbp = np.arange(len(ALLV)) >= len(VEC)   # BP groups appended with indices len(VEC)..
    W = {'full': np.zeros(len(ch)), 'h0': np.zeros(len(ch)), 'h1': np.zeros(len(ch))}
    W['full'][h0] = 0.5 / len(h0); W['full'][h1] = 0.5 / len(h1); W['h0'][h0] = 1 / len(h0); W['h1'][h1] = 1 / len(h1)
    RES = collections.defaultdict(list)
    for rep in range(3):
        perm = rng.permutation(len(VEC)); cal, sim = perm[::2], perm[1::2]; sim = sim[~np.isin(idx[sim], list(reft))]
        S = {}; NL = {}
        for k, w in W.items():
            s = ALLV @ w; s = (s - np.median(s[cal])) / s[cal].std(); S[k] = s; NL[k] = Null(s[cal])
        sc = {}
        G = sim[rng.integers(0, len(sim), (NM, 10, 3))]
        sc['honest'] = G
        for kk in (1, 2):
            g2 = G.copy(); g2[:, :, :kk] = len(VEC) + rng.integers(0, NB, (NM, 10, kk)); sc[f'beyond_policy_k{kk}'] = g2
        for n in (32, 64):
            dr = sim[rng.integers(0, len(sim), (NM, 10, 3 * n))]
            for nm, pr in (('length_refree', ln), ('gnorm_refree', gn)):
                sc[f'{nm}_x{n}'] = np.take_along_axis(dr, np.argsort(-pr[dr], axis=2)[:, :, :3], 2)
            sc[f'exact_leak_full_x{n}'] = np.take_along_axis(dr, np.argsort(-S['full'][dr], axis=2)[:, :, :3], 2)
        mu = float(S['full'][cal].mean())
        for nm, I in sc.items():
            X = S['full'][I].reshape(NM, -1) - mu; n = X.shape[1]
            single = gv(NL['full'], S['full'][I], ABON).mean()
            tt_ = X.mean(1) / (X.std(1, ddof=1) / np.sqrt(n)); T1 = (tt_ > tdist.isf(ABON, n - 1)).mean()
            Xb = S['full'][I].sum(2) - 3 * mu; tb = Xb.mean(1) / (Xb.std(1, ddof=1) / np.sqrt(Xb.shape[1])); T5 = (tb > tdist.isf(ABON, Xb.shape[1] - 1)).mean()
            T3 = np.mean([wilcoxon(x, alternative='greater').pvalue < ABON for x in X[:400]])
            RES[nm].append((float(single), float(T1), float(T3), float(T5)))
    names = ('v9_stouffer_max', 't_groups', 'wilcoxon_groups', 't_batch_sums')
    out[E] = {nm: {k: dict(mean=float(np.mean([r[j] for r in v])), min=float(np.min([r[j] for r in v])), max=float(np.max([r[j] for r in v]))) for j, k in enumerate(names)} for nm, v in RES.items()}
    print('== E', E, flush=True)
    for nm, r in out[E].items(): print(f'  {nm:22s}', ' '.join(f"{k}={x['mean']:.3f}[{x['min']:.2f},{x['max']:.2f}]" for k, x in r.items()), flush=True)
json.dump(out, open(HERE / 'data' / 'adv_cpu5.json', 'w'))
