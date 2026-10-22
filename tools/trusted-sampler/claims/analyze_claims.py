#!/usr/bin/env python3
"""Phase 3: per-batch scores, controls and metrics from claims/data/e<E>/dots-*.json. Aggregates/UIDs only."""
import glob, json, math, random, sys, collections
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
NPERM = 400

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])

def analyze(E, pattern='dots-[0-9]*.json', convert=None, nperm=None):
    D = [json.load(open(f)) for f in sorted(glob.glob(str(HERE / 'data' / f'e{E}' / pattern)))]
    NPERM_ = nperm or NPERM
    chunks = D[0]['chunks']; K = len(chunks)
    groups = [g for d in D for g in d['groups']]
    if convert: groups = [convert(g) for g in groups]
    byb = collections.defaultdict(list)
    for g in groups: byb[g['uid']].append(g)
    subjects = sorted({c['subject'] for c in chunks if c['subject']})
    ch = {(c['kind'], c['subject'], c['half']): [] for c in chunks}
    for k, c in enumerate(chunks): ch[(c['kind'], c['subject'], c['half'])].append(k)
    rng = np.random.default_rng(int(E))
    def score_vec(gs, perm=None, noise=False):
        """per-chunk first-order dJ_k per unit lr; perm: shuffle rewards within groups; noise: random rewards"""
        N = sum(len(g['rewards']) for g in gs); s = np.zeros(K)
        for g in gs:
            if not g['valid']: continue
            r = np.array(g['rewards'], float)
            if noise: r = rng.normal(0, r.std() or 1.0, len(r))
            elif perm: r = rng.permutation(r)
            A = r - r.mean(); s += A @ np.array(g['dots'])
        return s / (8192 * N)
    def alloc(shares, h, budget=6, exclude=()):
        order = [s for s, _ in sorted(shares.items(), key=lambda kv: -kv[1]) if s not in exclude]
        if not order: return []
        tot = sum(shares[s] for s in order); q = {s: budget * shares[s] / tot for s in order}
        n = {s: int(q[s]) for s in order}
        for s in sorted(order, key=lambda s: -(q[s] - n[s]))[:budget - sum(n.values())]: n[s] += 1
        ks = []
        for s in order:
            avail = ch.get(('subject', s, h), [])
            ks += avail[:n[s]]
        # fill shortfall from the remaining claimed subjects' chunks
        rest = [k for s in order for k in ch.get(('subject', s, h), []) if k not in ks]
        return (ks + rest)[:budget]
    rows = []
    for u, gs in byb.items():
        cnt = collections.Counter(g['subject'] for g in gs); tot = sum(cnt.values()); acc = 0; claim = {}
        for s, c in cnt.most_common():
            claim[s] = c / tot; acc += c
            if acc >= 0.8 * tot: break
        untouched = [s for s in subjects if s not in cnt] or [s for s, _ in sorted(cnt.items(), key=lambda kv: kv[1])[:2] if s not in claim]
        s = score_vec(gs)
        nullp = np.array([score_vec(gs, perm=True) for _ in range(NPERM_)])
        nulln = np.array([score_vec(gs, noise=True) for _ in range(100)])
        row = dict(uid=u, groups=len(gs), valid=sum(g['valid'] for g in gs), claim=claim, untouched=untouched)
        for h in (0, 1):
            est = {'claimed': alloc(claim, h), 'pool': ch[('pool', None, h)],
                   'wrong_claim': alloc({x: 1.0 for x in untouched}, h) if untouched else [],
                   'all_subjects': [k for x in subjects for k in ch.get(('subject', x, h), [])]}
            for name, ks in est.items():
                if not ks: row[f'{name}_{h}'] = None; continue
                row[f'{name}_{h}'] = float(s[ks].mean())
                row[f'{name}_{h}_nullsd'] = float(nullp[:, ks].mean(1).std())
                row[f'{name}_{h}_noisesd'] = float(nulln[:, ks].mean(1).std())
                row[f'{name}_{h}_nullmean'] = float(nullp[:, ks].mean(1).mean())
            row[f'_s{h}'] = s.tolist()
        rows.append(row)
    # hottest subject: highest mean per-subject score across batches, chosen on half h, applied on half 1-h
    hot = {}
    for h in (0, 1):
        m = {x: np.mean([np.mean(np.array(r[f'_s{h}'])[ch[('subject', x, h)]]) for r in rows]) for x in subjects if ch.get(('subject', x, h))}
        hot[h] = max(m, key=m.get)
    for r in rows:
        for h in (0, 1):
            ks = ch[("subject", hot[1 - h], h)][:6]; r[f'hot_{h}'] = float(np.array(r[f'_s{h}'])[ks].mean())
    def metric(name):
        a = np.array([r[f'{name}_0'] for r in rows if r.get(f'{name}_0') is not None and r.get(f'{name}_1') is not None])
        b = np.array([r[f'{name}_1'] for r in rows if r.get(f'{name}_0') is not None and r.get(f'{name}_1') is not None])
        sd0 = np.array([r[f'{name}_0_nullsd'] for r in rows if r.get(f'{name}_0') is not None and r.get(f'{name}_1') is not None])
        sd1 = np.array([r[f'{name}_1_nullsd'] for r in rows if r.get(f'{name}_0') is not None and r.get(f'{name}_1') is not None])
        z0, z1 = a / sd0, b / sd1; zc = (a + b) / np.sqrt(sd0 ** 2 + sd1 ** 2)
        diff_var = np.var(a - b, ddof=1) / 2  # per-half reference-noise variance
        tot_var = np.var(np.concatenate([a, b]), ddof=1)
        rel_half = float(np.corrcoef(a, b)[0, 1]); rel_full = 2 * rel_half / (1 + rel_half) if rel_half > -1 else None
        # pairs separable: |mean_i - mean_j| > 2*sqrt(var_i + var_j), per-batch variance = ref noise (from split-half) + label noise (null)
        m = (a + b) / 2; v = diff_var / 2 + (sd0 ** 2 + sd1 ** 2) / 4
        n = len(m); sep = 0; tot = 0
        for i in range(n):
            for j in range(i + 1, n):
                tot += 1; sep += abs(m[i] - m[j]) > 2 * math.sqrt(v[i] + v[j])
        return dict(batches=n, mean=float(m.mean()), sd_across_batches=float(m.std(ddof=1)),
                    median_null_sd=float(np.median(np.sqrt((sd0 ** 2 + sd1 ** 2) / 4))),
                    ref_noise_sd_per_half=float(math.sqrt(diff_var)),
                    snr_median_abs_z=float(np.median(np.abs(zc))), frac_abs_z_gt2=float(np.mean(np.abs(zc) > 2)),
                    n_pos_z_gt2=int(np.sum(zc > 2)), n_neg_z_lt_m2=int(np.sum(zc < -2)),
                    split_half_pearson=rel_half, split_half_spearman=spearman(a, b), spearman_brown_full=rel_full,
                    frac_pairs_separable=sep / tot if tot else None)
    res = {name: metric(name) for name in ('claimed', 'pool', 'wrong_claim', 'all_subjects')}
    # placebo summaries: z of null draws themselves is ~N(0,1) by construction; report shuffled vs noise sd ratio and null means
    res['placebo'] = dict(shuffled_label_mean_over_sd=float(np.median([abs(r['claimed_0_nullmean']) / r['claimed_0_nullsd'] for r in rows if r.get('claimed_0')])),
                          noise_sd_over_shuffled_sd=float(np.median([r['claimed_0_noisesd'] / r['claimed_0_nullsd'] for r in rows if r.get('claimed_0')])),
                          zero_step_identity_score=0.0)
    hc = np.array([[r['hot_0'] - r['claimed_0'], r['hot_1'] - r['claimed_1']] for r in rows if r.get('claimed_0') is not None])
    res['hottest_attack'] = dict(hot_subject_by_half={str(h): hot[h] for h in hot}, mean_gain_vs_true_claim=float(hc.mean()),
                                 frac_batches_gain=float((hc.mean(1) > 0).mean()),
                                 gain_over_null_sd=float(hc.mean() / np.median([r['claimed_0_nullsd'] for r in rows if r.get('claimed_0')])))
    wc = np.array([r['wrong_claim_0'] + r['wrong_claim_1'] - r['claimed_0'] - r['claimed_1'] for r in rows if r.get('wrong_claim_0') is not None and r.get('claimed_0') is not None]) / 2
    res['wrong_claim_vs_claimed'] = dict(mean_diff=float(wc.mean()), frac_wrong_higher=float((wc > 0).mean()))
    wcm = np.array([(r['wrong_claim_0'] + r['wrong_claim_1']) / 2 - (r['claimed_0'] + r['claimed_1']) / 2 for r in rows if r.get('wrong_claim_0') is not None and r.get('claimed_0') is not None])
    k = int((wcm < 0).sum()); n = len(wcm)
    res['wrong_claim_vs_claimed'].update(median_diff=float(np.median(wcm)), claimed_higher=k, n=n, sign_test_z=float((k - n / 2) / math.sqrt(n / 4)))
    # rank holding: top/bottom 3 by half-0 all-subjects score, their half-1 scores
    rk = sorted(rows, key=lambda r: -r['all_subjects_0'])
    res['rank_holdout'] = dict(top3_half0=[r['all_subjects_0'] for r in rk[:3]], top3_half1=[r['all_subjects_1'] for r in rk[:3]],
                               bottom3_half0=[r['all_subjects_0'] for r in rk[-3:]], bottom3_half1=[r['all_subjects_1'] for r in rk[-3:]],
                               top3_uids=[r['uid'] for r in rk[:3]], bottom3_uids=[r['uid'] for r in rk[-3:]])
    # overlap of batch tasks with the reference chunks (drift / leakage)
    reft = {t for c in chunks for t in c['tasks']}
    ov = np.array([sum(g['index'] in reft for g in byb[r['uid']]) for r in rows]); sc = np.array([r['all_subjects_0'] + r['all_subjects_1'] for r in rows])
    res['reference_overlap'] = dict(reference_tasks=len(reft), batches_with_overlap=int((ov > 0).sum()), mean_overlap_groups=float(ov.mean()),
                                    spearman_overlap_vs_score=spearman(ov, sc) if ov.std() > 0 else None)
    res['claimed_vs_pool_corr'] = spearman(np.array([r['claimed_0'] + r['claimed_1'] for r in rows]), np.array([r['pool_0'] + r['pool_1'] for r in rows]))
    claimsize = collections.Counter(len(r['claim']) for r in rows)
    res['meta'] = dict(batches=len(rows), groups=sum(r['groups'] for r in rows), valid_groups=sum(r['valid'] for r in rows),
                       claim_sizes=dict(sorted(claimsize.items())), chunks=K, informative_by_subject=D[0]['informative_by_subject'],
                       subspace=D[0]['subspace'], timings=[d['timings'] for d in D])
    ranked = sorted([r for r in rows if r.get('claimed_0') is not None], key=lambda r: -(r['claimed_0'] + r['claimed_1']))
    res['top3_uids'] = [r['uid'] for r in ranked[:3]]; res['bottom3_uids'] = [r['uid'] for r in ranked[-3:]]
    for r in rows:
        for h in (0, 1): r.pop(f'_s{h}')
    return res, rows

if __name__ == '__main__':
    out = {}
    for E in sys.argv[1].split(','):
        res, rows = analyze(E); out[E] = dict(summary=res, rows=rows)
        print(E, json.dumps(res, indent=1, default=str)[:6000])
    json.dump(out, open(HERE / 'data' / 'claims_results.json', 'w'), default=str)
