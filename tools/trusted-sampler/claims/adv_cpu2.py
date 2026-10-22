#!/usr/bin/env python3
"""v9 adversarial pass 2 (CPU): reference-free norm-proxy cherry-picking, subject tuning evaluated out-of-sample (+ stratified-null defense),
payoff in payment units V = sum_g (s_g - m_E) for gated miners (DESIGN-v9 s4.8), sybil payoff, Spearman transfer, copies/replay scores."""
import glob, gzip, json, sys, collections
from pathlib import Path
import numpy as np
from scipy.stats import norm, binom, spearmanr
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import grader, grad_real
from plant_dots import key
rew = lambda c, L: grader.overlong_reward(c, L)
rng = np.random.default_rng(99); NM = 2000; ABON = 0.01 / 210
T = json.load(open(HERE.parent / 'data/original-math7496.tasks.json'))
def gvec(rws, dots):
    r = np.asarray(rws, float); return ((r - r.mean()) @ np.asarray(dots)) / (8192 * len(r))
class Null:
    def __init__(self, cal, S=3, n=200000):
        self.m = np.sort(cal[rng.integers(0, len(cal), (n, S))].max(1)); med, sd = np.median(self.m), self.m.std(); self.q = float((self.m > med + 2 * sd).mean())
    def p(self, M): return np.clip(1.0 - np.searchsorted(self.m, M, side='left') / len(self.m), 0.5 / len(self.m), 1.0)
def gate(null, M):
    p = null.p(M); B = M.shape[1]; z = norm.isf(p).sum(1) / np.sqrt(B); hits = (p < null.q).sum(1)
    return (z > norm.isf(ABON)) & (binom.sf(hits - 1, B, null.q) < ABON), z
def summarize(null, Sc, m_E):   # Sc [miners, B, 3] group scores -> gate rate, mean payout V (in units of honest group SD), mean z
    g, z = gate(null, Sc.max(2)); V = np.where(g, np.clip((Sc - m_E).sum((1, 2)), 0, None), 0.0)
    return dict(gated=float(g.mean()), stouffer01=float((z > norm.isf(0.01)).mean()), V=float(V.mean()))
out = {}
for E in ('62', '64', '66'):
    D = [json.load(open(f)) for f in sorted(glob.glob(str(HERE / 'data' / f'e{E}' / 'dots-real-*.json')))]; ch = D[0]['chunks']
    sub = [k for k, c in enumerate(ch) if c['kind'] == 'subject']; h0 = [k for k in sub if ch[k]['half'] == 0]; h1 = [k for k in sub if ch[k]['half'] == 1]
    W = {'full': np.zeros(len(ch)), 'h0': np.zeros(len(ch)), 'h1': np.zeros(len(ch))}
    W['full'][h0] = 0.5 / len(h0); W['full'][h1] = 0.5 / len(h1); W['h0'][h0] = 1 / len(h0); W['h1'][h1] = 1 / len(h1)
    reft = {i for c in ch for i in c['tasks']}
    R = {(r['miner'], r['slot']): r for r in (json.loads(l) for l in gzip.open(HERE / 'data/real' / f'real-{E}.jsonl.gz', 'rt'))}
    SK = {}
    for f in glob.glob(str(HERE / 'data' / 'gnorm' / f'e{E}' / 'sk-*.npz')):
        z = np.load(f); SK.update(zip(z['keys'].tolist(), z['sk']))
    V, idx, gn, ln = [], [], [], []
    for d in D:
        for g in d['groups']:
            r = R[(g['miner'], g['slot'])]; rws = [rew(1.0 if c == 'positive' else 0.0, L) for c, L in zip(g['claimed_cls'], g['lens'])]
            V.append(gvec(rws, g['dots'])); idx.append(r['index']); ln.append(np.mean(g['lens']))
            sks = [SK.get(key(r['index'], np.asarray(x['out'], dtype=np.int32))) for x in r['rollouts']]
            A = np.array(rws) - np.mean(rws)
            gn.append(float(np.linalg.norm(A @ np.stack(sks))) / (8192 * len(rws)) if all(s is not None for s in sks) else np.nan)
    V = np.array(V); idx = np.array(idx); gn = np.array(gn); ln = np.array(ln); S = {k: V @ w for k, w in W.items()}
    subj = np.array([T[i]['data']['subject'] for i in idx]); ins = np.isin(idx, list(reft))
    perm = rng.permutation(len(V)); cal, sim = perm[::2], perm[1::2]; sim = sim[~ins[sim]]
    sd = {k: S[k][cal].std() for k in S}; mE = {k: float(np.median(S[k][cal])) for k in S}
    Sn = {k: (S[k] - mE[k]) / sd[k] for k in S}   # in honest-SD units
    nul = {k: Null(Sn[k][cal]) for k in S}; res = {}
    # Spearman across random disjoint chunk splits
    sp = []
    for _ in range(50):
        pm = rng.permutation(sub); a, b = pm[:len(sub) // 2], pm[len(sub) // 2:]; sp.append(spearmanr(V[:, a].mean(1), V[:, b].mean(1))[0])
    res['spearman_random_split'] = dict(mean=float(np.mean(sp)), sd=float(np.std(sp)), h0_h1=float(spearmanr(S['h0'], S['h1'])[0]))
    res['norm_vs_score'] = dict(spearman_gnorm_full=float(spearmanr(gn[~np.isnan(gn)], S['full'][~np.isnan(gn)])[0]), spearman_len_full=float(spearmanr(ln, S['full'])[0]),
                                spearman_gnorm_absfull=float(spearmanr(gn[~np.isnan(gn)], np.abs(Sn['full'][~np.isnan(gn)]))[0]))
    B = 10; pay = {}
    def draw(pool, k3=3): return pool[rng.integers(0, len(pool), (NM, B, k3))]
    pay['honest'] = summarize(nul['full'], Sn['full'][draw(sim)], 0.0)
    for n in (8, 32, 64):
        dr = draw(sim, 3 * n)
        for nm, proxy, evalk in (('proxy_salted', 'h0', 'h1'), ('exact_leak', 'h1', 'h1'), ('exact_leak_full', 'full', 'full'), ('gnorm_refree', 'gnorm', 'full'), ('length_refree', 'len', 'full')):
            pr = {'gnorm': np.nan_to_num(gn, nan=-1), 'len': ln}.get(proxy, S.get(proxy)); top = np.take_along_axis(dr, np.argsort(-pr[dr], axis=2)[:, :, :3], 2)
            pay[f'{nm}_x{n}'] = summarize(nul[evalk], Sn[evalk][top], 0.0)
    # subject tuning: choose the subject with the best h0 mean (attacker's proxy), evaluate on h1 only; defense = per-subject standardization
    sm = {s: float(Sn['h0'][sim][subj[sim] == s].mean()) for s in set(subj)}; best = max(sm, key=sm.get); tuned = sim[subj[sim] == best]
    pay['subject_tuned_h1'] = summarize(nul['h1'], Sn['h1'][draw(tuned)], 0.0); pay['subject_tuned_h1']['subject'] = best
    st = Sn['h1'].copy()
    for s in set(subj):
        mk = subj == s; cm = mk.copy(); cm[sim] = False; cm &= np.isin(np.arange(len(V)), cal)
        if cm.sum() > 5: st[mk] = (Sn['h1'][mk] - np.median(Sn['h1'][cm])) / Sn['h1'][cm].std()
    nst = Null(st[cal]); pay['subject_tuned_h1_stratified'] = summarize(nst, st[draw(tuned)], 0.0)
    pay['honest_h1_stratified'] = summarize(nst, st[draw(sim)], 0.0)
    # beyond-policy (bo16) payoff for reference
    P = HERE / 'data' / 'planted' / f'e{E}'; cache = {}
    for f in glob.glob(str(P / 'cache-*.json')): cache.update(json.load(open(f)))
    teach = json.load(open(P / 'teacher.json')); tz = np.load(P / 'teacher.npz'); to = tz['offsets']; tt = tz['tokens']; base, meta = grad_real.load_ref(E)
    byref = collections.defaultdict(list)
    for x in meta:
        if x['kind'] == 'ref': byref[x['index']].append(x)
    cor = collections.defaultdict(list)
    for kk, c in enumerate(teach['correct']):
        if c['j'] < 16: cor[c['index']].append((c['j'], tt[to[kk]:to[kk + 1]]))
    bp = []
    for i, cs in cor.items():
        cs = sorted(cs, key=lambda c: c[0])[:2]; wr = [x for x in byref[i] if x['outcome'] != 'positive'][:6]
        toks = [c[1] for c in cs] + [x['tokens'] for x in wr]; rws = [rew(1.0, len(c[1])) for c in cs] + [rew(0.0, x['len']) for x in wr]
        dts = [cache.get(key(i, t)) for t in toks]
        if all(d is not None for d in dts): bp.append(((gvec(rws, dts) @ W['full']) - mE['full']) / sd['full'])
    bp = np.array(bp); res['bp_group_z'] = dict(mean=float(bp.mean()), median=float(np.median(bp)))
    for k in (1, 2):
        Sc = Sn['full'][draw(sim)].copy(); Sc[:, :, :k] = bp[rng.integers(0, len(bp), (NM, B, k))]; pay[f'beyond_policy_k{k}'] = summarize(nul['full'], Sc, 0.0)
    # sybil payoff: K bp groups per window split over h hotkeys (B=10 batches each), total expected V across hotkeys
    sy = {}
    for K in (10, 20, 30):
        for h in (1, 2, 3, 5, 10):
            per = K // h; tot = []
            for rep in range(300):
                Sc = Sn['full'][sim[rng.integers(0, len(sim), (h, B, 3))]].copy()
                for hk in range(h):
                    for j in range(min(per, B * 3)): Sc[hk, j % B, j // B] = bp[rng.integers(0, len(bp))]
                g, _ = gate(nul['full'], Sc.max(2)); tot.append(float(np.where(g, np.clip((Sc).sum((1, 2)), 0, None), 0).sum()))
            sy[f'K{K}_h{h}'] = float(np.mean(tot))
    res['sybil_total_V'] = sy; res['payoff_B10'] = pay
    # copies / replay (if dots exist)
    A = HERE / 'data' / 'adv' / f'e{E}'; cp = {}
    for nm in ('copies', 'distill', 'replay62'):
        if not (A / f'{nm}.json').exists(): continue
        dd = {}
        for f in glob.glob(str(A / f'dots-{nm}-*.json')): dd.update(json.load(open(f)))
        if not dd: continue
        m = json.load(open(A / f'{nm}.json'))['meta']; z = np.load(A / f'{nm}.npz'); o = z['offsets']; tk = z['tokens']
        ks = [key(x['index'], tk[o[n]:o[n + 1]]) for n, x in enumerate(m)]
        if nm == 'replay62':
            by = collections.defaultdict(list)
            for x, k in zip(m, ks): by[x['index']].append((x, k))
            zs, zin = [], []
            for i, xs in by.items():
                if any(k not in dd for _, k in xs): continue
                sc = ((gvec([x['reward'] for x, _ in xs], [dd[k] for _, k in xs]) @ W['full']) - mE['full']) / sd['full']; (zin if xs[0][0]['in_dst_ref'] else zs).append(sc)
            cp[nm] = dict(n=len(zs) + len(zin), mean_z=float(np.mean(zs + zin)), mean_z_not_in_ref=float(np.mean(zs)) if zs else None, mean_z_in_ref=float(np.mean(zin)) if zin else None,
                          frac_above_bp_median=float(np.mean(np.array(zs + zin) > np.median(bp))), bp_same_epoch_mean_z=float(bp.mean()))
            continue
        wr = {i: [x for x in byref[i] if x['outcome'] != 'positive'][:6] for i in byref}
        byv = collections.defaultdict(lambda: collections.defaultdict(list))
        for n, (x, k) in enumerate(zip(m, ks)):
            if x['outcome'] == 'positive' and k in dd: byv[x['variant']][x['index']].append((n, k))
        for var, tasks in byv.items():
            zs, jac = [], []
            for i, xs in tasks.items():
                w6 = wr.get(i, [])
                if len(w6) < 6: continue
                wk = [cache.get(key(i, x['tokens'])) for x in w6]
                if any(v is None for v in wk): continue
                c2 = xs[:2]; rws = [rew(1.0, m[n]['len']) for n, _ in c2] + [rew(0.0, x['len']) for x in w6]
                zs.append(((gvec(rws, [dd[k] for _, k in c2] + wk) @ W['full']) - mE['full']) / sd['full']); jac += [m[n]['jac5'] for n, _ in c2 if m[n]['jac5'] is not None]
            if zs:
                Sc = Sn['full'][draw(sim)].copy(); zz = np.array(zs); Sc[:, :, 0] = zz[rng.integers(0, len(zz), (NM, B))]
                cp[f'{nm}_{var}'] = dict(n_groups=len(zs), mean_z=float(np.mean(zs)), median_z=float(np.median(zs)), jac5_median=float(np.median(jac)) if jac else None,
                                         jac5_p10=float(np.quantile(jac, 0.1)) if jac else None, k1_gated=summarize(nul['full'], Sc, 0.0)['gated'])
        # correct rate per variant
        cnt = collections.Counter((x['variant'], x['outcome'] == 'positive') for x in m)
        cp[f'{nm}_correct_rate'] = {v: cnt[(v, True)] / max(1, cnt[(v, True)] + cnt[(v, False)]) for v in {x['variant'] for x in m}}
    res['copies_replay'] = cp
    out[E] = res
    print('== E', E, json.dumps({k: v for k, v in res.items() if k != 'payoff_B10'}), flush=True)
    print(' payoff', json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in pay.items()}), flush=True)
json.dump(out, open(HERE / 'data' / 'adv_cpu2.json', 'w'))
