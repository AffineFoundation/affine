#!/usr/bin/env python3
"""Score planted batches with the identical all-subjects estimator; compare to placebo null and to the real-miner population."""
import glob, json, sys, collections, hashlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from plant_dots import key
EPOCHS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['62', '64']; NPERM = 400
def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg): return None
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))
real_res = json.load(open(HERE / 'data/claims_real_results.json'))
out = {}; typemeans = {}
for E in EPOCHS:
    D = HERE / 'data' / 'planted' / f'e{E}'; spec = json.load(open(D / 'spec.json'))
    RD = {}; chunks = None
    for f in glob.glob(str(HERE / 'data' / f'e{E}' / 'dots-real-*.json')):
        d = json.load(open(f)); chunks = d['chunks']
        for g in d['groups']: RD[(g['miner'], g['slot'])] = g['dots']
    RG = {(g['uid'], g['index']): g['dots'] for f in glob.glob(str(HERE / 'data' / f'e{E}' / 'dots-[0-9]*.json')) for g in json.load(open(f))['groups'] if g['valid']}
    cache = {}
    for f in glob.glob(str(D / 'cache-*.json')): cache.update(json.load(open(f)))
    m = json.load(open(D / 'new.json')); z = np.load(D / 'new.npz'); o = z['offsets']; tk = z['tokens']
    NEW = [cache.get(key(x['index'], tk[o[n]:o[n + 1]])) for n, x in enumerate(m['meta'])]
    K = len(chunks); h0 = [k for k, c in enumerate(chunks) if c['kind'] == 'subject' and c['half'] == 0]; h1 = [k for k, c in enumerate(chunks) if c['kind'] == 'subject' and c['half'] == 1]
    rng = np.random.default_rng(int(E))
    def dots(r):
        if r['src'] == 'real': return RD[tuple(r['key'][:2])][r['key'][2]]
        if r['src'] == 'regen': return RG[(r['key'][0], r['key'][1])][r['key'][2]]
        return NEW[r['key']]
    rows = []; missing = 0
    for b in spec['batches']:
        try: gs = [(np.array([r['reward'] for r in g['rollouts']]), np.array([dots(r) for r in g['rollouts']])) for g in b['groups']]
        except (TypeError, KeyError, ValueError): missing += 1; continue
        if any(d.ndim != 2 for _, d in gs): missing += 1; continue
        N = sum(len(r) for r, _ in gs)
        def sc(perm):
            s = np.zeros(K)
            for r, d in gs:
                rr = rng.permutation(r) if perm else r; s += (rr - rr.mean()) @ d
            s /= 8192 * N; return (s[h0].mean() + s[h1].mean()) / 2, s[h0].mean(), s[h1].mean()
        s, a0, a1 = sc(False); null = np.array([sc(True)[0] for _ in range(NPERM)])
        rows.append(dict(type=b['type'], id=b['id'], groups=len(gs), score=float(s), half0=float(a0), half1=float(a1), null_sd=float(null.std()), z_null=float(s / null.std()) if null.std() > 0 else 0.0))
    pop = [r for r in real_res['epochs'][E]['rows']]
    ps = np.array([(r['all_subjects_0'] + r['all_subjects_1']) / 2 for r in pop]); pm, psd = float(np.median(ps)), float(ps.std())
    pz = np.array([(r['all_subjects_0'] + r['all_subjects_1']) / np.sqrt(r['all_subjects_0_nullsd'] ** 2 + r['all_subjects_1_nullsd'] ** 2) for r in pop])
    for r in rows: r['z_pop'] = (r['score'] - pm) / psd; r['pct_of_real'] = float((ps < r['score']).mean())
    T = collections.defaultdict(list)
    for r in rows: T[r['type']].append(r)
    summ = {}
    for t, rs in sorted(T.items()):
        sc_ = np.array([r['score'] for r in rs]); zn = np.array([r['z_null'] for r in rs]); zp = np.array([r['z_pop'] for r in rs])
        summ[t] = dict(n=len(rs), mean_score=float(sc_.mean()), sd_within_type=float(sc_.std(ddof=1)) if len(rs) > 1 else None, mean_z_null=float(zn.mean()),
                       frac_z_gt2=float((zn > 2).mean()), frac_z_lt_m2=float((zn < -2).mean()), mean_z_pop=float(zp.mean()),
                       n_distinguishable_from_honest_pop=int((np.abs(zp) > 2).sum()), median_pct_of_real=float(np.median([r['pct_of_real'] for r in rs])),
                       split_half_corr_within_type=float(np.corrcoef([r['half0'] for r in rs], [r['half1'] for r in rs])[0, 1]) if len(rs) > 2 else None)
    honest = [r['score'] for r in T['1_honest_resampled' if '1_honest_resampled' in T else '1_honest']] + list(ps)
    good = [r['score'] for t in T if t.startswith(('7', '8')) for r in T[t]]
    bad = [r['score'] for t in T if t.startswith(('2', '3', '4')) for r in T[t]]
    aucs = dict(good_vs_honest=auc(good, honest), bad_vs_honest=auc(bad, honest), good_vs_bad=auc(good, bad),
                **{f'{t}_vs_honest': auc([r['score'] for r in T[t]], honest) for t in T if not t.startswith('1')})
    ranking = sorted(summ, key=lambda t: -summ[t]['mean_score'])
    out[E] = dict(types=summ, auc=aucs, ranking=ranking, real_pop=dict(n=len(ps), median=pm, sd=psd, frac_z_gt2=float((pz > 2).mean()), frac_z_lt_m2=float((pz < -2).mean())),
                  missing_batches=missing, rows=rows, spec_meta={k: v for k, v in spec.items() if k != 'batches'})
    typemeans[E] = {t: summ[t]['mean_score'] for t in summ}
    print('== E', E, 'real pop', out[E]['real_pop'], 'missing', missing)
    for t in ranking:
        s = summ[t]; print(f"{t:28s} n {s['n']:2d} mean {s['mean_score']:+.2e} sdW {s['sd_within_type'] or 0:.2e} z_null {s['mean_z_null']:+.2f} z>2 {s['frac_z_gt2']:.1f} z<-2 {s['frac_z_lt_m2']:.1f} z_pop {s['mean_z_pop']:+.2f} distinct {s['n_distinguishable_from_honest_pop']} pct {s['median_pct_of_real']:.2f}")
    print('AUC', json.dumps({k: (round(v, 2) if v is not None else None) for k, v in aucs.items()}))
if len(typemeans) == 2:
    a, b = EPOCHS; ts = sorted(set(typemeans[a]) & set(typemeans[b]))
    x = np.array([typemeans[a][t] for t in ts]); y = np.array([typemeans[b][t] for t in ts])
    rk = lambda v: np.argsort(np.argsort(v))
    out['cross_epoch_type_means'] = dict(types=ts, spearman=float(np.corrcoef(rk(x), rk(y))[0, 1]), pearson=float(np.corrcoef(x, y)[0, 1]))
    print('cross-epoch type-mean Spearman', out['cross_epoch_type_means'])
json.dump(out, open(HERE / 'data' / 'planted_results.json', 'w'))
