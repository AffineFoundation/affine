#!/usr/bin/env python3
"""Four-way comparison from results-combined/*.jsonl. usage: combined_analyze.py results-combined/ [--json out]"""
import glob, json, math, sys, statistics as st
import numpy as np
recs = [json.loads(l) for f in glob.glob(sys.argv[1].rstrip('/') + '/*.jsonl') for l in open(f)]
def cost(T):
    g = lambda *k: sum(T.get(x, 0) for x in k)
    base = g('load_B', 'gen_B', 'verify_B')
    return {'baseline_indep': base + g('swap_to_A', 'gen_A_indep', 'verify_A_indep'),
            'crn': base + g('swap_to_A', 'gen_A_crn', 'verify_A_crn'),
            'logprob_only': base + g('lp_Bsamp_on_B', 'swap_to_A', 'lp_Bsamp_on_A'),
            'crn+logprob': base + g('lp_Bsamp_on_B', 'swap_to_A', 'gen_A_crn', 'verify_A_crn', 'lp_Bsamp_on_A', 'lp_Asamp_on_A', 'swap_to_B', 'lp_Asamp_on_B')}
out = {}
for step in sorted({r['step'] for r in recs}):
    rs = [r for r in recs if r['step'] == step]; P = [t for r in rs for t in r['tasks']]; n = len(P)
    col = lambda k: np.array([t[k] for t in P])
    D = {k: col(k) for k in ('d_ind', 'd_crn', 'd_lp', 'd_rev')}
    X = np.vstack([D['d_crn'], D['d_lp'], D['d_rev']]); C = np.cov(X); w = np.linalg.solve(C, np.ones(3)); w /= w.sum()
    D['d_comb'] = w @ X
    X2 = np.vstack([D['d_crn'], D['d_lp']]); C2 = np.cov(X2); w2 = np.linalg.solve(C2, np.ones(2)); w2 /= w2.sum()
    D['d_crn_lp2'] = w2 @ X2
    costs = [cost(r['timings_s']) for r in rs]; cmin = {k: st.mean(c[k] for c in costs) / 60 for k in costs[0]}
    est = {}
    for name, key, ck in (('baseline_indep', 'd_ind', 'baseline_indep'), ('crn', 'd_crn', 'crn'), ('logprob_only', 'd_lp', 'logprob_only'),
                          ('crn+logprob', 'd_comb', 'crn+logprob'), ('crn+logprob(2-way: crn,fwd-IS)', 'd_crn_lp2', 'crn+logprob'),
                          ('reverse-IS (A gen + lp)', 'd_rev', 'crn+logprob')):
        v = D[key]; sd = float(v.std(ddof=1))
        est[name] = dict(delta=float(v.mean()), se_pooled=sd / math.sqrt(n), se_at_K128=sd / math.sqrt(128),
                         gpu_min_per_K128=cmin[ck], var_x_cost_rel=None)
    b = est['baseline_indep']['se_at_K128'] ** 2 * est['baseline_indep']['gpu_min_per_K128']
    for e in est.values(): e['var_x_cost_rel'] = e['se_at_K128'] ** 2 * e['gpu_min_per_K128'] / b
    sn = [t['s_norm'] for t in P if t['s_norm'] is not None]
    out[step] = dict(before=rs[0]['before'][:8], after=rs[0]['after'][:8], runs=len(rs), tasks=n, estimators=est,
                     comb_weights=dict(zip(['crn', 'fwd_IS', 'rev_IS'], map(float, w))),
                     corr_cB_cA_crn=float(np.corrcoef(col('cB'), col('cA_crn'))[0, 1]), corr_cB_cA_ind=float(np.corrcoef(col('cB'), col('cA_ind'))[0, 1]),
                     s_norm=dict(tasks=len(sn), frac=len(sn) / n, mean=st.mean(sn), se=st.stdev(sn) / math.sqrt(len(sn)), z=st.mean(sn) / (st.stdev(sn) / math.sqrt(len(sn)))),
                     ess_fwd_median=float(np.median(col('ess_fwd'))), ess_rev_median=float(np.median(col('ess_rev'))),
                     mean_seq_dlogp_B=float(col('mean_dlogp_seq_B').mean()),
                     timings_mean_s={k: round(st.mean(r['timings_s'].get(k, 0) for r in rs), 1) for k in rs[0]['timings_s']})
print(json.dumps(out, indent=1))
if '--json' in sys.argv: json.dump(out, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1)
