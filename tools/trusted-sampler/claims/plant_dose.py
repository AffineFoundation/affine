#!/usr/bin/env python3
"""Dose-response / mixtures / detection threshold on stored gradient-dots (+ grad-norm sketches when present). CPU only.
Score of a batch = mean over all-subject chunks (both halves) of sum_g A_g . D_g / (8192 N)  (identical to the pipeline).
Detection: z_pop = (score - median(real)) / sd(real) > 2, real = this epoch's real miner batches (uid != None)."""
import glob, gzip, json, sys, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import grader, grad_real
from plant_dots import key
EP = sys.argv[1].split(',') if len(sys.argv) > 1 else ['62', '64', '66']
rng = np.random.default_rng(11); NDRAW = 400
rew = lambda c, L: grader.overlong_reward(c, L)
def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))
out = {}
for E in EP:
    D = [json.load(open(f)) for f in sorted(glob.glob(str(HERE / 'data' / f'e{E}' / 'dots-real-*.json')))]; ch = D[0]['chunks']
    sub = [k for k, c in enumerate(ch) if c['kind'] == 'subject']; h0 = [k for k in sub if ch[k]['half'] == 0]; h1 = [k for k in sub if ch[k]['half'] == 1]
    w = np.zeros(len(ch)); w[h0] = 0.5 / len(h0); w[h1] = 0.5 / len(h1)          # score = w . s
    R = {(r['miner'], r['slot']): r for r in (json.loads(l) for l in gzip.open(HERE / 'data/real' / f'real-{E}.jsonl.gz', 'rt'))}
    # sketches (optional)
    SK = {}
    for f in glob.glob(str(HERE / 'data' / 'gnorm' / f'e{E}' / 'sk-*.npz')):
        z = np.load(f); SK.update(zip(z['keys'].tolist(), z['sk']))
    def G(rewards, dots, sks):   # group: (sum A d [K], sum A sk [DS] or None, n)
        r = np.asarray(rewards); A = r - r.mean(); dv = A @ np.asarray(dots)
        sv = (A @ np.stack(sks)) if sks is not None and all(s is not None for s in sks) else None
        return dict(d=dv, s=sv, n=len(r))
    honest = []; byu = collections.defaultdict(list)
    for d in D:
        for g in d['groups']:
            r = R[(g['miner'], g['slot'])]; rw = [rew(1.0 if c == 'positive' else 0.0, L) for c, L in zip(g['claimed_cls'], g['lens'])]
            sks = [SK.get(key(r['index'], np.asarray(x['out'], dtype=np.int32))) for x in r['rollouts']] if SK else None
            gg = G(rw, g['dots'], sks); honest.append(gg)
            if g['uid'] is not None: byu[g['uid']].append(gg)
    def score(gs):
        N = sum(g['n'] for g in gs); s = sum(g['d'] for g in gs) / (8192 * N); raw = float(w @ s)
        cos = None
        if all(g['s'] is not None for g in gs):
            nb = np.linalg.norm(sum(g['s'] for g in gs)) / (8192 * N); cos = raw / nb if nb > 0 else 0.0
        return raw, cos
    pop = [score(gs) for gs in byu.values()]
    praw = np.array([p[0] for p in pop]); med, sd = float(np.median(praw)), float(praw.std())
    pcos = np.array([p[1] for p in pop if p[1] is not None]); hascos = len(pcos) == len(pop)
    # group-level honest distribution for the max-group rule
    gsc = np.array([float(w @ g['d']) / (8192 * g['n']) for g in honest]); gm, gs_ = float(np.median(gsc)), float(gsc.std())
    def maxg(gs): return max((float(w @ g['d']) / (8192 * g['n']) - gm) / gs_ for g in gs)
    popmax = np.array([maxg(gs) for gs in byu.values()]); mm, ms = float(np.median(popmax)), float(popmax.std())
    # beyond-policy group pools by best-of-N (teacher correct with sample index j < N, first <=2, + wrong base samples)
    P = HERE / 'data' / 'planted' / f'e{E}'; base, meta = grad_real.load_ref(E)
    cache = {}
    for f in glob.glob(str(P / 'cache-*.json')): cache.update(json.load(open(f)))
    teach = json.load(open(P / 'teacher.json')); tz = np.load(P / 'teacher.npz'); to = tz['offsets']; tt = tz['tokens']
    byref = collections.defaultdict(list)
    for x in meta:
        if x['kind'] == 'ref': byref[x['index']].append(x)
    cor = collections.defaultdict(list)
    for k, c in enumerate(teach['correct']): cor[c['index']].append((c['j'], tt[to[k]:to[k + 1]]))
    Ns = [n for n in (4, 8, 16, 40) if n <= teach['samples']]
    pools = {}
    for Nn in Ns:
        pl = []
        for i, cs in cor.items():
            cs = sorted([c for c in cs if c[0] < Nn], key=lambda c: c[0])[:2]
            if not cs: continue
            wr = [x for x in byref[i] if x['outcome'] != 'positive'][:6]
            toks = [c[1] for c in cs] + [x['tokens'] for x in wr]; rws = [rew(1.0, len(c[1])) for c in cs] + [rew(0.0, x['len']) for x in wr]
            dts = [cache.get(key(i, t)) for t in toks]
            if any(d is None for d in dts): continue
            sks = [SK.get(key(i, np.asarray(t, dtype=np.int32))) for t in toks] if SK else None
            pl.append(G(rws, dts, sks))
        pools[Nn] = pl
    res = dict(real_pop=dict(n=len(pop), median=med, sd=sd), teacher=dict(tasks=teach['tasks'], samples=teach['samples']),
               pool_sizes={str(k): len(v) for k, v in pools.items()}, dose={}, mixture={}, size={})
    def evaluate(batches):
        sc = [score(b) for b in batches]; raw = np.array([s[0] for s in sc]); zr = (raw - med) / sd
        r = dict(n=len(batches), mean_z_pop=float(zr.mean()), detect_rate=float((zr > 2).mean()), auc_vs_real=auc(raw, praw))
        if hascos and all(s[1] is not None for s in sc):
            c = np.array([s[1] for s in sc]); zc = (c - np.median(pcos)) / pcos.std()
            r.update(cos_mean_z_pop=float(zc.mean()), cos_detect_rate=float((zc > 2).mean()), cos_auc_vs_real=auc(c, pcos))
        mx = np.array([maxg(b) for b in batches]); zm = (mx - mm) / ms
        r.update(maxgroup_mean_z_pop=float(zm.mean()), maxgroup_detect_rate=float((zm > 2).mean()), maxgroup_auc_vs_real=auc(mx, popmax))
        return r
    for Nn, pl in pools.items():   # 1) pure best-of-N batches of 9 groups (10 disjoint if possible, else resampled)
        if len(pl) < 9: res['dose'][str(Nn)] = dict(n=0, pool=len(pl)); continue
        idx = rng.permutation(len(pl)); bs = [[pl[j] for j in idx[b * 9:(b + 1) * 9]] for b in range(min(10, len(pl) // 9))]
        while len(bs) < 10: bs.append([pl[j] for j in rng.choice(len(pl), 9, replace=False)])
        res['dose'][str(Nn)] = dict(evaluate(bs), pool=len(pl), disjoint=min(10, len(pl) // 9))
    # 2) mixtures: honest batch of 9 resampled real groups with k replaced by best-of-N groups; 3) other batch sizes
    hon_idx = np.arange(len(honest))
    for Nn, pl in pools.items():
        if len(pl) < 5: continue
        for k in range(0, 10):
            bs = []
            for _ in range(NDRAW):
                h = [honest[j] for j in rng.choice(hon_idx, 9 - k, replace=False)]; bp = [pl[j] for j in rng.choice(len(pl), min(k, len(pl)), replace=False)]
                bs.append(h + bp)
            res['mixture'][f'bo{Nn}_k{k}'] = evaluate(bs)
    Nbest = max(pools, key=lambda n: len(pools[n]) if n == 16 else -1) if 16 in pools else max(pools)
    for B in (3, 6, 9, 18, 36):   # honest resampled null at size B (not the real-miner pop: that only exists at B~9)
        hb = [[honest[j] for j in rng.choice(hon_idx, B, replace=False)] for _ in range(NDRAW)]
        hs = np.array([score(b)[0] for b in hb]); hm, hsd = float(np.median(hs)), float(hs.std())
        hmx = np.array([maxg(b) for b in hb]); hmm, hms = float(np.median(hmx)), float(hmx.std())
        row = dict(honest_sd=hsd)
        for k in (1, 2):
            if k >= B: continue
            for Nn in (16, 40):
                if Nn not in pools or len(pools[Nn]) < 2: continue
                bs = [[honest[j] for j in rng.choice(hon_idx, B - k, replace=False)] + [pools[Nn][j] for j in rng.choice(len(pools[Nn]), k, replace=False)] for _ in range(NDRAW)]
                z = (np.array([score(b)[0] for b in bs]) - hm) / hsd; zm = (np.array([maxg(b) for b in bs]) - hmm) / hms
                row[f'k{k}_bo{Nn}'] = dict(power_raw=float((z > 2).mean()), power_maxgroup=float((zm > 2).mean()), mean_z_raw=float(z.mean()), mean_z_maxgroup=float(zm.mean()))
        res['size'][str(B)] = row
    # reference-size scaling: max-group power for k=1,2 (B=9) using one reference half (~half the tasks) vs both halves
    wb = w.copy(); rs = {}
    for name, ww in (('half0', np.where(np.isin(np.arange(len(ch)), h0), 1.0 / len(h0), 0.0)), ('half1', np.where(np.isin(np.arange(len(ch)), h1), 1.0 / len(h1), 0.0)), ('both', wb)):
        w[:] = ww
        g2 = np.array([float(w @ g['d']) / (8192 * g['n']) for g in honest]); gm, gs_ = float(np.median(g2)), float(g2.std())
        hb = [[honest[j] for j in rng.choice(hon_idx, 9, replace=False)] for _ in range(NDRAW)]
        hmx = np.array([maxg(b) for b in hb]); hmm, hms = float(np.median(hmx)), float(hmx.std())
        row = {}
        for k in (1, 2):
            for Nn in (16, 40):
                if Nn not in pools: continue
                bs = [[honest[j] for j in rng.choice(hon_idx, 9 - k, replace=False)] + [pools[Nn][j] for j in rng.choice(len(pools[Nn]), k, replace=False)] for _ in range(NDRAW)]
                zm = (np.array([maxg(b) for b in bs]) - hmm) / hms; row[f'k{k}_bo{Nn}'] = float((zm > 2).mean())
            # single-group separation: AUC of one beyond-policy group vs one honest group
        row['group_auc_bo40'] = auc([float(w @ g['d']) / (8192 * g['n']) for g in pools.get(40, pools[max(pools)])], g2)
        rs[name] = row
    w[:] = wb; g2 = np.array([float(w @ g['d']) / (8192 * g['n']) for g in honest]); gm, gs_ = float(np.median(g2)), float(g2.std())
    res['ref_scaling'] = rs
    # minimal detectable k at 80% power (batch of 9) per best-of-N and rule
    md = {}
    for Nn in pools:
        for rule, kk in (('raw', 'detect_rate'), ('cos', 'cos_detect_rate'), ('maxgroup', 'maxgroup_detect_rate')):
            ks = [k for k in range(10) if res['mixture'].get(f'bo{Nn}_k{k}', {}).get(kk, 0) >= 0.8]
            md[f'bo{Nn}_{rule}'] = min(ks) if ks else None
    res['min_detectable_k_of_9'] = md
    res['fpr_k0'] = {f'bo{Nn}': res['mixture'].get(f'bo{Nn}_k0', {}).get('detect_rate') for Nn in pools}
    out[E] = res
    print('== E', E, 'pop', res['real_pop'], 'pools', res['pool_sizes'], 'cos' if hascos else 'no-cos', flush=True)
    for Nn, r in res['dose'].items(): print(' dose bo', Nn, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}))
    for k in (0, 1, 2, 3, 5, 9):
        for Nn in pools:
            r = res['mixture'].get(f'bo{Nn}_k{k}')
            if r: print(f' mix bo{Nn} k={k}', json.dumps({kk: round(v, 2) for kk, v in r.items() if isinstance(v, float)}))
    print(' min k', md); print(' ref_scaling', json.dumps(res['ref_scaling'])); print(' size', json.dumps(res['size']))
json.dump(out, open(HERE / 'data' / 'dose_results.json', 'w'))
