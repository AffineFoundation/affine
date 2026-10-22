#!/usr/bin/env python3
"""Band-filter power test (CPU) on data/adv/e64/band.json: reference = 2..6/8 tasks (current) vs 1..7/8 (no band filter: adds 1/8 and 7/8 tasks)."""
import glob, gzip, json, sys, collections
from pathlib import Path
import numpy as np
from scipy.stats import norm, binom
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import grader, grad_real
from plant_dots import key
rew = lambda c, L: grader.overlong_reward(c, L); E = sys.argv[1] if len(sys.argv) > 1 else '64'
bd = json.load(open(HERE / 'data' / 'adv' / f'e{E}' / 'band.json')); X = bd['dots']; xch = bd['chunks']
D = [json.load(open(f)) for f in sorted(glob.glob(str(HERE / 'data' / f'e{E}' / 'dots-real-*.json')))]; ch = D[0]['chunks']
sub = [k for k, c in enumerate(ch) if c['kind'] == 'subject']
R = {(r['miner'], r['slot']): r for r in (json.loads(l) for l in gzip.open(HERE / 'data/real' / f'real-{E}.jsonl.gz', 'rt'))}
G = {(g['miner'], g['slot']): g for d in D for g in d['groups']}
def vec(A, d0, dx): return np.concatenate([(A @ np.array(d0))[sub], A @ np.array(dx)]) / (8192 * len(A))
rows, kind = [], []
for mk in map(tuple, bd['groups']):
    g = G[mk]; r = R[mk]; A = np.array([rew(1.0 if c == 'positive' else 0.0, L) for c, L in zip(g['claimed_cls'], g['lens'])]); A = A - A.mean()
    dx = [X[key(r['index'], np.asarray(x['out'], dtype=np.int32))] for x in r['rollouts']]; rows.append(vec(A, g['dots'], dx)); kind.append('h')
P = HERE / 'data' / 'planted' / f'e{E}'; cache = {}
for f in glob.glob(str(P / 'cache-*.json')): cache.update(json.load(open(f)))
teach = json.load(open(P / 'teacher.json')); tz = np.load(P / 'teacher.npz'); to = tz['offsets']; tt = tz['tokens']; _, meta = grad_real.load_ref(E)
byref = collections.defaultdict(list)
for x in meta:
    if x['kind'] == 'ref': byref[x['index']].append(x)
cor = collections.defaultdict(list)
for k, c in enumerate(teach['correct']):
    if c['j'] < 16: cor[c['index']].append((c['j'], k))
for i, cs in cor.items():
    ks = [k for _, k in sorted(cs)[:2]]; wr = [x for x in byref[i] if x['outcome'] != 'positive'][:6]
    toks = [tt[to[k]:to[k + 1]] for k in ks] + [x['tokens'] for x in wr]; r = np.array([rew(1.0, len(t)) for t in toks[:len(ks)]] + [rew(0.0, x['len']) for x in wr])
    kk = [key(i, t) for t in toks]
    if all(k in cache and k in X for k in kk): rows.append(vec(r - r.mean(), [cache[k] for k in kk], [X[k] for k in kk])); kind.append('bp')
M = np.array(rows); kind = np.array(kind); n0 = len(sub)
h = lambda lst, half: [j for j, k in enumerate(lst) if (ch[k]['half'] if lst is sub else xch[k]['half']) == half]
s0 = [j for j in range(n0) if ch[sub[j]]['half'] == 0]; s1 = [j for j in range(n0) if ch[sub[j]]['half'] == 1]
x0 = [n0 + j for j, c in enumerate(xch) if c['half'] == 0]; x1 = [n0 + j for j, c in enumerate(xch) if c['half'] == 1]
def W(a0, a1):
    w = np.zeros(M.shape[1]); w[a0] = .5 / len(a0); w[a1] = .5 / len(a1); return w
rng = np.random.default_rng(11); out = {}
for nm, w in (('band_2_6', W(s0, s1)), ('band_1_7', W(s0 + x0, s1 + x1)), ('only_1or7', W(x0, x1))):
    s = M @ w; hs = s[kind == 'h']; mu, sd = np.median(hs), hs.std(); z = (s - mu) / sd
    res = []
    for rep in range(10):   # gate sim, B=10, 3-group batches, k=1 BP
        perm = rng.permutation(np.where(kind == 'h')[0]); cal, sim = perm[::2], perm[1::2]
        nl = np.sort(z[cal][rng.integers(0, len(cal), (200000, 3))].max(1)); q = float((nl > np.median(nl) + 2 * nl.std()).mean())
        pv = lambda Mx: np.clip(1 - np.searchsorted(nl, Mx) / len(nl), 0.5 / len(nl), 1)
        bpi = np.where(kind == 'bp')[0]; I = sim[rng.integers(0, len(sim), (2000, 10, 3))]; I[:, :, 0] = bpi[rng.integers(0, len(bpi), (2000, 10))]
        for lab, II in (('bp', I), ('honest', sim[rng.integers(0, len(sim), (2000, 10, 3))])):
            p = pv(z[II].max(2)); Z = norm.isf(p).sum(1) / np.sqrt(10); hh = (p < q).sum(1); A_ = 0.01 / 210
            res.append((lab, float(((Z > norm.isf(A_)) & (binom.sf(hh - 1, 10, q) < A_)).mean()), float((Z > norm.isf(0.01)).mean())))
    out[nm] = dict(bp_mean_z=float(z[kind == 'bp'].mean()), bp_median_z=float(np.median(z[kind == 'bp'])), n_bp=int((kind == 'bp').sum()), n_h=int((kind == 'h').sum()),
                   gated_bp_k1=float(np.mean([g for l, g, _ in res if l == 'bp'])), st01_bp=float(np.mean([s_ for l, _, s_ in res if l == 'bp'])),
                   gated_honest=float(np.mean([g for l, g, _ in res if l == 'honest'])), st01_honest=float(np.mean([s_ for l, _, s_ in res if l == 'honest'])))
out['n_extra_tasks'] = bd['n_extra_tasks']; out['n_extra_chunks'] = len(xch); out['n_orig_subject_chunks'] = n0
print(json.dumps(out, indent=1)); json.dump(out, open(HERE / 'data' / 'adv_band.json', 'w'), indent=1)
