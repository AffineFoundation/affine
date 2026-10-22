#!/usr/bin/env python3
"""v2: adds split-half replication statistics and repeats over 3 CAL/SIM splits.
Defenses against variance-inflation (reference-free length / gradient-norm selection) and proxy cherry-picking (CPU).
Group statistic variants (all calibrated on the out-of-sample CAL half, 3-group batches, B=10, gate = Stouffer AND binomial at 0.01/210):
  raw   : s_g (v9 as proposed)
  stud  : t_g = s_g / sd_perm(s_g), sd_perm^2 = sum A_i^2 * sum (x_i - xbar)^2 / (n-1)  (exact within-group label-permutation SD; x_i = d_i . w)
  sign  : raw, but the per-miner test counts NET hits = (#batches with max-group p<q) - (#batches with min-group lower-tail p<q); sign test.
  stud+sign : both."""
import glob, gzip, json, sys, collections
from pathlib import Path
import numpy as np
from scipy.stats import norm, binom
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import grader, grad_real
from plant_dots import key
rew = lambda c, L: grader.overlong_reward(c, L)
rng = np.random.default_rng(5); NM = 2000; ABON = 0.01 / 210; B = 10
def stats(rws, xs):
    r = np.asarray(rws, float); x = np.asarray(xs, float); A = r - r.mean(); n = len(r); s = float(A @ x) / (8192 * n)
    sdp = np.sqrt((A ** 2).sum() * ((x - x.mean()) ** 2).sum() / (n - 1)) / (8192 * n); return s, (s / sdp if sdp > 0 else 0.0)
class Null:
    def __init__(self, cal, n=200000):
        g = cal[rng.integers(0, len(cal), (n, 3))]; self.mx = np.sort(g.max(1)); self.mn = np.sort(g.min(1))
        self.q = float((self.mx > np.median(self.mx) + 2 * self.mx.std()).mean())
    def p_hi(self, M): return np.clip(1.0 - np.searchsorted(self.mx, M, side='left') / len(self.mx), 0.5 / len(self.mx), 1 - 0.5 / len(self.mx))
    def p_lo(self, M): return np.clip(np.searchsorted(self.mn, M, side='right') / len(self.mn), 0.5 / len(self.mn), 1 - 0.5 / len(self.mn))
def gate(nl, Sc, signed):
    ph = nl.p_hi(Sc.max(2)); z = norm.isf(ph).sum(1) / np.sqrt(B); hits = (ph < nl.q).sum(1)
    if not signed: return float(((z > norm.isf(ABON)) & (binom.sf(hits - 1, B, nl.q) < ABON)).mean())
    pl = nl.p_lo(Sc.min(2)); lo = (pl < nl.q).sum(1); net = hits - lo
    # sign test on hi vs lo hits among batches with any tail hit, plus the Stouffer gate on the hi tail minus the lo tail
    zs = (norm.isf(ph) - norm.isf(pl)).sum(1) / np.sqrt(2 * B)
    return float(((zs > norm.isf(ABON)) & (binom.sf(hits - 1, B, nl.q) < ABON) & (net >= 2)).mean())
out = {}
for E in ('62', '64', '66'):
    D = [json.load(open(f)) for f in sorted(glob.glob(str(HERE / 'data' / f'e{E}' / 'dots-real-*.json')))]; ch = D[0]['chunks']
    sub = [k for k, c in enumerate(ch) if c['kind'] == 'subject']; h0 = [k for k in sub if ch[k]['half'] == 0]; h1 = [k for k in sub if ch[k]['half'] == 1]
    w = np.zeros(len(ch)); w[h0] = 0.5 / len(h0); w[h1] = 0.5 / len(h1); w0 = np.zeros(len(ch)); w0[h0] = 1 / len(h0)
    reft = {i for c in ch for i in c['tasks']}
    R = {(r['miner'], r['slot']): r for r in (json.loads(l) for l in gzip.open(HERE / 'data/real' / f'real-{E}.jsonl.gz', 'rt'))}
    SK = {}
    for f in glob.glob(str(HERE / 'data' / 'gnorm' / f'e{E}' / 'sk-*.npz')):
        z = np.load(f); SK.update(zip(z['keys'].tolist(), z['sk']))
    raw, stu, prox, gn, ln, idx, VEC = [], [], [], [], [], [], []
    for d in D:
        for g in d['groups']:
            r = R[(g['miner'], g['slot'])]; rws = [rew(1.0 if c == 'positive' else 0.0, L) for c, L in zip(g['claimed_cls'], g['lens'])]
            dt = np.array(g['dots']); s, t = stats(rws, dt @ w); raw.append(s); stu.append(t); prox.append(stats(rws, dt @ w0)[0]); idx.append(r['index']); ln.append(np.mean(g['lens']))
            Ar = np.array(rws) - np.mean(rws); VEC.append((Ar @ dt) / (8192 * len(rws)))
            sks = [SK.get(key(r['index'], np.asarray(x['out'], dtype=np.int32))) for x in r['rollouts']]; A = np.array(rws) - np.mean(rws)
            gn.append(float(np.linalg.norm(A @ np.stack(sks))) if all(v is not None for v in sks) else -1.0)
    raw, stu, prox, gn, ln, idx, VEC = map(np.array, (raw, stu, prox, gn, ln, idx, VEC))
    q4 = [[k for n_, k in enumerate(sorted(sub, key=lambda k: (ch[k]['subject'], ch[k]['half'], ch[k]['c']))) if n_ % 4 == j] for j in range(4)]
    def derived(Vm, cal_):   # returns dict of standardized stats for chunk-vector matrix Vm, standardization from CAL rows
        out_ = {}
        def stdz(x, ref): return (x - np.median(ref)) / ref.std()
        a0 = Vm[:, h0].mean(1); a1 = Vm[:, h1].mean(1); c0 = VR[cal_][:, h0].mean(1); c1 = VR[cal_][:, h1].mean(1)
        out_['min_halves'] = np.minimum(stdz(a0, c0), stdz(a1, c1))
        qs = [stdz(Vm[:, q].mean(1), VR[cal_][:, q].mean(1)) for q in q4]; out_['median_quarters'] = np.median(np.stack(qs), 0)
        return out_
    VR = VEC
    SPL = []
    for rep in range(3):
        perm = rng.permutation(len(raw)); cal, sim = perm[::2], perm[1::2]; sim = sim[~np.isin(idx[sim], list(reft))]; SPL.append((cal, sim))
    ALL = []
    for rep, (cal, sim) in enumerate(SPL):
        Z = {'raw': (raw - np.median(raw[cal])) / raw[cal].std(), 'stud': (stu - np.median(stu[cal])) / stu[cal].std()}
        Z.update(derived(VEC, cal))
        NL = {k: Null(v[cal]) for k, v in Z.items()}
        # beyond-policy groups
        P = HERE / 'data' / 'planted' / f'e{E}'; cache = {}
        for f in glob.glob(str(P / 'cache-*.json')): cache.update(json.load(open(f)))
        teach = json.load(open(P / 'teacher.json')); tz = np.load(P / 'teacher.npz'); to = tz['offsets']; tt = tz['tokens']; base, meta = grad_real.load_ref(E)
        byref = collections.defaultdict(list)
        for x in meta:
            if x['kind'] == 'ref': byref[x['index']].append(x)
        cor = collections.defaultdict(list)
        for kk, c in enumerate(teach['correct']):
            if c['j'] < 16: cor[c['index']].append((c['j'], tt[to[kk]:to[kk + 1]]))
        bpr, bps, BPV = [], [], []
        for i, cs in cor.items():
            cs = sorted(cs, key=lambda c: c[0])[:2]; wr = [x for x in byref[i] if x['outcome'] != 'positive'][:6]
            toks = [c[1] for c in cs] + [x['tokens'] for x in wr]; rws = [rew(1.0, len(c[1])) for c in cs] + [rew(0.0, x['len']) for x in wr]
            dts = [cache.get(key(i, t)) for t in toks]
            if all(v is not None for v in dts):
                s, t = stats(rws, np.array(dts) @ w); bpr.append(s); bps.append(t); Ar = np.array(rws) - np.mean(rws); BPV.append((Ar @ np.array(dts)) / (8192 * len(rws)))
        BP = {'raw': (np.array(bpr) - np.median(raw[cal])) / raw[cal].std(), 'stud': (np.array(bps) - np.median(stu[cal])) / stu[cal].std()}
        BP.update(derived(np.array(BPV), cal))
        res = dict(bp_mean_z={k: float(v.mean()) for k, v in BP.items()}, single_group_auc={})
        for k in Z:
            a, b_ = BP[k], Z[k][sim]; res['single_group_auc'][k] = float((a[:, None] > b_[None, :]).mean())
        scen = {}
        def run(name, sel):
            row = {}
            for k in Z:
                Sc = sel(k)
                row[k] = gate(NL[k], Sc, False)
            scen[name] = row
        def honest(k): return Z[k][sim[rng.integers(0, len(sim), (NM, B, 3))]]
        run('honest', honest)
        for kk in (1, 2):
            def bp_(k, kk=kk):
                Sc = honest(k).copy(); Sc[:, :, :kk] = BP[k][rng.integers(0, len(BP[k]), (NM, B, kk))]; return Sc
            run(f'beyond_policy_k{kk}', bp_)
        for n in (32, 64):
            dr = sim[rng.integers(0, len(sim), (NM, B, 3 * n))]
            for nm, pr in (('length_refree', ln), ('gnorm_refree', gn), ('proxy_salted_h0', prox)):
                top = np.take_along_axis(dr, np.argsort(-pr[dr], axis=2)[:, :, :3], 2)
                run(f'{nm}_x{n}', lambda k, top=top: Z[k][top])
            top = np.take_along_axis(dr, np.argsort(-Z['raw'][dr], axis=2)[:, :, :3], 2); run(f'exact_leak_raw_x{n}', lambda k, top=top: Z[k][top])
            top2 = np.take_along_axis(dr, np.argsort(-Z['stud'][dr], axis=2)[:, :, :3], 2); run(f'exact_leak_stud_x{n}', lambda k, top=top2: Z[k][top])
        ALL.append((res, scen))
    res = ALL[0][0]; keys = list(ALL[0][1]); stats_ = list(ALL[0][1][keys[0]])
    res['gate_rates_B10'] = {sc: {k: dict(mean=float(np.mean([A[1][sc][k] for A in ALL])), min=float(np.min([A[1][sc][k] for A in ALL])), max=float(np.max([A[1][sc][k] for A in ALL]))) for k in stats_} for sc in keys}
    out[E] = res
    print('== E', E, json.dumps(res['bp_mean_z']), json.dumps(res['single_group_auc']), flush=True)
    for sc, r in res['gate_rates_B10'].items(): print(f'  {sc:24s}', ' '.join(f"{k}={v['mean']:.2f}[{v['min']:.2f},{v['max']:.2f}]" for k, v in r.items()), flush=True)
json.dump(out, open(HERE / 'data' / 'adv_cpu3.json', 'w'))
