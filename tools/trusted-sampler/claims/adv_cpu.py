#!/usr/bin/env python3
"""v9 adversarial simulations on stored gradient-dots (CPU). Per epoch, every honest real group g has a chunk vector
v_g[k] = sum_i A_i d_i[k] / (8192 n). Scores: full = pipeline weighting (both halves), h0/h1 = one half only (independent task draws).
v9 test: 3-group batches, batch stat = max-group score, p from out-of-sample CAL null at S=3, per-miner Stouffer over B batches;
gates: Stouffer at alpha=0.01, Bonferroni alpha=0.01/210, binomial gate at 0.01/210, BH-FDR q=0.05 over 210 miners."""
import glob, gzip, json, sys, collections
from pathlib import Path
import numpy as np
from scipy.stats import norm, binom, spearmanr, pearsonr
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import grader, grad_real, grad_claims as gc
from plant_dots import key
rew = lambda c, L: grader.overlong_reward(c, L)
rng = np.random.default_rng(7); NM = 2000; NMINERS = 210; A0 = 0.01; ABON = A0 / NMINERS
T = json.load(open(HERE.parent / 'data/original-math7496.tasks.json'))
def gvec(rws, dots):
    r = np.asarray(rws, float); return ((r - r.mean()) @ np.asarray(dots)) / (8192 * len(r))
class Null:
    def __init__(self, cal, S=3, n=200000):
        self.m = np.sort(cal[rng.integers(0, len(cal), (n, S))].max(1)); med, sd = np.median(self.m), self.m.std(); self.q = float((self.m > med + 2 * sd).mean()); self.S = S
    def p(self, M): return np.clip(1.0 - np.searchsorted(self.m, M, side='left') / len(self.m), 0.5 / len(self.m), 1.0)
def gates(null, M):   # M [miners, B] batch max scores -> flag rates for each gate
    p = null.p(M); B = M.shape[1]; z = norm.isf(p).sum(1) / np.sqrt(B); hits = (p < null.q).sum(1); pb = binom.sf(hits - 1, B, null.q)
    return dict(stouffer01=float((z > norm.isf(A0)).mean()), stouffer_bon=float((z > norm.isf(ABON)).mean()), binom_bon=float((pb < ABON).mean()),
                both_bon=float(((z > norm.isf(ABON)) & (pb < ABON)).mean()), mean_z=float(z.mean())), z
def bh(pvals, q=0.05):
    n = len(pvals); o = np.argsort(pvals); thr = q * np.arange(1, n + 1) / n; ok = pvals[o] <= thr
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0; fl = np.zeros(n, bool); fl[o[:k]] = True; return fl
out = {}
for E in ('62', '64', '66'):
    D = [json.load(open(f)) for f in sorted(glob.glob(str(HERE / 'data' / f'e{E}' / 'dots-real-*.json')))]; ch = D[0]['chunks']
    sub = [k for k, c in enumerate(ch) if c['kind'] == 'subject']; h0 = [k for k in sub if ch[k]['half'] == 0]; h1 = [k for k in sub if ch[k]['half'] == 1]
    W = {'full': np.zeros(len(ch)), 'h0': np.zeros(len(ch)), 'h1': np.zeros(len(ch))}
    W['full'][h0] = 0.5 / len(h0); W['full'][h1] = 0.5 / len(h1); W['h0'][h0] = 1 / len(h0); W['h1'][h1] = 1 / len(h1)
    reftasks = {i for c in ch for i in c['tasks']}; R = {(r['miner'], r['slot']): r for r in (json.loads(l) for l in gzip.open(HERE / 'data/real' / f'real-{E}.jsonl.gz', 'rt'))}
    V, idx, pas, gn = [], [], [], []
    for d in D:
        for g in d['groups']:
            r = R[(g['miner'], g['slot'])]; rws = [rew(1.0 if c == 'positive' else 0.0, L) for c, L in zip(g['claimed_cls'], g['lens'])]
            V.append(gvec(rws, g['dots'])); idx.append(r['index']); pas.append(np.mean([c == 'positive' for c in g['claimed_cls']]))
    V = np.array(V); idx = np.array(idx); pas = np.array(pas); S = {k: V @ w for k, w in W.items()}
    res = dict(n_groups=len(V))
    # ---- 1. transfer: correlation of group scores across independent reference draws
    r01p = pearsonr(S['h0'], S['h1'])[0]; r01s = spearmanr(S['h0'], S['h1'])[0]
    sp = []
    for _ in range(50):   # random chunk splits within subjects (both halves pooled)
        perm = rng.permutation(sub); a, b = perm[:len(sub) // 2], perm[len(sub) // 2:]; sp.append(pearsonr(V[:, a].mean(1), V[:, b].mean(1))[0])
    rel = 2 * r01p / (1 + r01p)
    res['transfer'] = dict(pearson_h0_h1=float(r01p), spearman_h0_h1=float(r01s), random_split_pearson_mean=float(np.mean(sp)), random_split_sd=float(np.std(sp)),
                           reliability_full=float(rel), max_proxy_corr_full=float(np.sqrt(max(rel, 0))))
    # ---- in-sample: honest groups whose task IS a reference-chunk task
    ins = np.isin(idx, list(reftasks)); zf = (S['full'] - np.median(S['full'])) / S['full'].std()
    res['in_sample'] = dict(n_in_ref=int(ins.sum()), frac=float(ins.mean()), mean_z_in=float(zf[ins].mean()) if ins.any() else None, mean_z_out=float(zf[~ins].mean()),
                            p_top5pct_in=float((zf[ins] > np.quantile(zf, 0.95)).mean()) if ins.any() else None)
    # ---- v9 cherry-picking: proxy = h0 (attacker's own reference draw) or exact private (leak), private = h1 (+ in-sample tasks excluded)
    perm = rng.permutation(len(V)); cal, sim = perm[::2], perm[1::2]; sim = sim[~ins[sim]]
    nul = {k: Null(S[k][cal]) for k in ('h1', 'full')}
    cp = {}
    for proxy, private in (('h0', 'h1'), ('h1', 'h1'), ('full', 'full')):
        for n in (1, 2, 4, 8, 16, 32, 64):
            for B in (6, 10):
                draw = sim[rng.integers(0, len(sim), (NM, B, 3 * n))]; pr = S[proxy][draw]
                top = np.take_along_axis(draw, np.argsort(-pr, axis=2)[:, :, :3], 2); M = S[private][top].max(2)
                g_, _ = gates(nul[private], M); cp[f'{proxy}->{private}_n{n}_B{B}'] = g_
    res['cherry_pick'] = cp
    # ---- 2. leak: submit groups on reference tasks (in-sample), exact rule known
    lk = {}
    if ins.sum() >= 10:
        refg = np.where(ins)[0]
        for B in (6, 10):
            M = S['full'][refg[rng.integers(0, len(refg), (NM, B, 3))]].max(2); lk[f'refTasks_B{B}'], _ = gates(nul['full'], M)
            dr = refg[rng.integers(0, len(refg), (NM, B, 3 * 8))]; top = np.take_along_axis(dr, np.argsort(-S['full'][dr], axis=2)[:, :, :3], 2)
            lk[f'refTasks_select8_B{B}'], _ = gates(nul['full'], S['full'][top].max(2))
    res['leak'] = lk
    # ---- 6. gradient-norm inflation: planted garbage sequences relabeled WRONG (honest labels) inside real groups
    P = HERE / 'data' / 'planted' / f'e{E}'; spec = json.load(open(P / 'spec.json')); cache = {}
    for f in glob.glob(str(P / 'cache-*.json')): cache.update(json.load(open(f)))
    m = json.load(open(P / 'new.json')); z = np.load(P / 'new.npz'); o = z['offsets']; tk = z['tokens']
    RD = {(g['miner'], g['slot']): g['dots'] for d in D for g in d['groups']}
    gar = []; gar1 = []
    for b in spec['batches']:
        if not b['type'].startswith('4'): continue
        for g in b['groups']:
            dts, rws, isg = [], [], []
            for r in g['rollouts']:
                if r['src'] == 'new':
                    dts.append(cache[key(m['meta'][r['key']]['index'], tk[o[r['key']]:o[r['key'] + 1]])]); rws.append(0.0); isg.append(True)
                else:
                    dts.append(RD[tuple(r['key'][:2])][r['key'][2]]); rws.append(r['reward']); isg.append(False)
            if np.std(rws) > 0: gar.append(gvec(rws, dts) @ W['full'])
    gar = np.array(gar)
    g6 = dict(n_groups=len(gar), mean_z=float(((gar - np.median(S['full'])) / S['full'].std()).mean()) if len(gar) else None,
              sd_ratio=float(gar.std() / S['full'].std()) if len(gar) else None)
    if len(gar) >= 5:
        for B in (6, 10):
            M = np.stack([np.concatenate([gar[rng.integers(0, len(gar), (NM, B, 1))], S['full'][sim[rng.integers(0, len(sim), (NM, B, 2))]]], 2).max(2)])[0]
            g6[f'1garbage_per_batch_B{B}'], _ = gates(nul['full'], M)
            g6[f'3garbage_per_batch_B{B}'], _ = gates(nul['full'], gar[rng.integers(0, len(gar), (NM, B, 3))].max(2))
    res['gnorm_inflation'] = g6
    # ---- 7. easy-task mining / subject tuning
    e7 = {}
    easy = sim[pas[sim] >= 0.75]; hard = sim[pas[sim] <= 0.25]
    subj = np.array([T[i]['data']['subject'] for i in idx]); sm = {s: float(S['h0'][sim][subj[sim] == s].mean()) for s in set(subj)}
    best = max(sm, key=sm.get); tuned = sim[subj[sim] == best]
    for nm, pool in (('easy_ge75', easy), ('hard_le25', hard), (f'subject_tuned', tuned)):
        if len(pool) < 10: continue
        e7[nm] = dict(n=len(pool), **gates(nul['full'], S['full'][pool[rng.integers(0, len(pool), (NM, 10, 3))]].max(2))[0])
    e7['tuned_subject'] = best; e7['subject_h0_means_z'] = {s: v / S['h0'].std() for s, v in sm.items()}
    res['easy_subject'] = e7
    # ---- multiple testing: 3-group batches, k beyond-policy per batch, B=3..12 at alpha=0.01/210 and BH-FDR(0.05) with 210 miners
    bp = []
    teach = json.load(open(P / 'teacher.json')); tz = np.load(P / 'teacher.npz'); to = tz['offsets']; tt = tz['tokens']; base, meta = grad_real.load_ref(E)
    byref = collections.defaultdict(list)
    for x in meta:
        if x['kind'] == 'ref': byref[x['index']].append(x)
    cor = collections.defaultdict(list)
    for kk, c in enumerate(teach['correct']):
        if c['j'] < 16: cor[c['index']].append((c['j'], tt[to[kk]:to[kk + 1]]))
    for i, cs in cor.items():
        cs = sorted(cs, key=lambda c: c[0])[:2]; wr = [x for x in byref[i] if x['outcome'] != 'positive'][:6]
        toks = [c[1] for c in cs] + [x['tokens'] for x in wr]; rws = [rew(1.0, len(c[1])) for c in cs] + [rew(0.0, x['len']) for x in wr]
        dts = [cache.get(key(i, t)) for t in toks]
        if all(d is not None for d in dts): bp.append(gvec(rws, dts) @ W['full'])
    bp = np.array(bp); mt = {}
    for k in (1, 2):
        for B in range(3, 13):
            g = sim[rng.integers(0, len(sim), (NM, B, 3))]; Ssc = S['full'][g]; Ssc[:, :, :k] = bp[rng.integers(0, len(bp), (NM, B, k))]
            gg, zz = gates(nul['full'], Ssc.max(2))
            # BH: 210 miners, 10 contributors + 200 honest, repeated 200 times
            bhp, bhf = [], []
            for rep in range(200):
                gh = S['full'][sim[rng.integers(0, len(sim), (200, B, 3))]].max(2); _, zh = gates(nul['full'], gh)
                zc = zz[rng.integers(0, NM, 10)]; pv = norm.sf(np.concatenate([zc, zh])); fl = bh(pv)
                bhp.append(fl[:10].mean()); bhf.append(fl[10:].sum() / max(1, fl.sum()))
            mt[f'k{k}_B{B}'] = dict(**gg, bh_power=float(np.mean(bhp)), bh_fdp=float(np.mean(bhf)))
    hon = {}
    for B in (3, 6, 10, 12):
        g_, _ = gates(nul['full'], S['full'][sim[rng.integers(0, len(sim), (20000, B, 3))]].max(2)); hon[f'B{B}'] = g_
    res['multiple_testing'] = dict(contributors=mt, honest_fp=hon, bp_pool=len(bp))
    # ---- 4. sybil: K beyond-policy groups per window; spread over h hotkeys, each with B=10 batches of 3 groups
    sy = {}
    for K in (5, 10, 20):
        for h in (1, 2, 5, 10, 20):
            if h > K: continue
            per = K // h; Bk = 10; flags = []
            for rep in range(400):
                g = S['full'][sim[rng.integers(0, len(sim), (h, Bk, 3))]].copy()
                for hk in range(h):
                    for j in range(min(per, Bk * 3)):
                        g[hk, j % Bk, j // Bk] = bp[rng.integers(0, len(bp))]
                ga, zz = gates(nul['full'], g.max(2)); flags.append((zz > norm.isf(ABON)).sum())
            sy[f'K{K}_h{h}'] = dict(per_hotkey=per, exp_flagged_hotkeys=float(np.mean(flags)), p_any=float(np.mean(np.array(flags) > 0)))
    res['sybil'] = sy
    out[E] = res
    print('== E', E, json.dumps(res['transfer']), json.dumps(res['in_sample']), flush=True)
    for kk in ('h0->h1', 'h1->h1', 'full->full'):
        print(' cherry', kk, {n: (round(cp[f'{kk}_n{n}_B10']['stouffer01'], 2), round(cp[f'{kk}_n{n}_B10']['both_bon'], 3), round(cp[f'{kk}_n{n}_B10']['mean_z'], 2)) for n in (1, 2, 4, 8, 16, 32, 64)})
    print(' leak', json.dumps(lk)); print(' gnorm', json.dumps(g6)); print(' easy', json.dumps({k: v for k, v in e7.items() if k != 'subject_h0_means_z'}))
    print(' mt', {k: (round(v['stouffer_bon'], 2), round(v['both_bon'], 2), round(v['bh_power'], 2), round(v['bh_fdp'], 3)) for k, v in mt.items()}); print(' honestfp', json.dumps(hon))
    print(' sybil', json.dumps(sy))
json.dump(out, open(HERE / 'data' / 'adv_cpu.json', 'w'))
