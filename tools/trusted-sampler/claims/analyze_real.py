#!/usr/bin/env python3
"""Phase 3 on REAL miner rollouts: same estimators/controls as analyze_claims.py plus label audit, staleness,
label-corruption control, cross-epoch UID stability, regenerated-vs-real comparison. Aggregates/UIDs only."""
import gzip, json, sys, glob, collections, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import grader
from analyze_claims import analyze, spearman
EPOCHS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['62', '63', '64', '65', '66']
rew = lambda cls, L: [grader.overlong_reward(1.0 if c == 'positive' else 0.0, l) for c, l in zip(cls, L)]
def conv(g):
    return dict(uid=g['uid'], index=g['index'], subject=g['subject'], valid=True, rewards=rew(g['claimed_cls'], g['lens']), dots=g['dots'])
out = {}; scores = {}
for E in EPOCHS:
    fs = sorted(glob.glob(str(HERE / 'data' / f'e{E}' / 'dots-real-*.json')))
    if len(fs) < 4: print('skip', E, len(fs)); continue
    res, rows = analyze(E, 'dots-real-*.json', conv)
    D = [json.load(open(f)) for f in fs]; G = [g for d in D for g in d['groups']]; chunks = D[0]['chunks']
    R = [json.loads(l) for l in gzip.open(HERE / 'data/real' / f'real-{E}.jsonl.gz', 'rt')]
    rg = json.load(open(HERE / 'data/real' / f'regrade-{E}.json'))
    # label audit (our verifier vs miner-claimed classification); per-batch error rate
    ours = {(d['miner'], d['slot']): d['ours'] for d in rg['docs']}
    err_b = collections.defaultdict(lambda: [0, 0]); tot = wrong = flip_corr = 0; indet = 0
    for r in R:
        o = ours[(r['miner'], r['slot'])]
        for x, k in zip(r['rollouts'], o):
            tot += 1; indet += k == 'indeterminate'
            bad = k != 'indeterminate' and ((x['cls'] == 'positive') != (k == 'positive'))
            wrong += bad; err_b[r['uid']][0] += bad; err_b[r['uid']][1] += 1
    sall = {r['uid']: r['all_subjects_0'] + r['all_subjects_1'] for r in rows}
    eb = np.array([err_b[u][0] / err_b[u][1] for u in sall]); sv = np.array(list(sall.values()))
    res['label_audit'] = dict(rollouts=tot, claimed_label_wrong=wrong, frac_wrong=wrong / tot, indeterminate=indet,
                              confusion=rg['confusion'], batches_with_any_error=int((eb > 0).sum()),
                              spearman_score_vs_error_rate=spearman(eb, sv) if eb.std() > 0 else None)
    res['staleness'] = dict(built_on=dict(collections.Counter(r['built_on'] for r in R)), base=D[0]['base'][:16],
                            frac_on_base=float(np.mean([r['built_on'] == D[0]['base'][:16] for r in R])), spearman_score_vs_staleness=None)
    res['prompt_matches_ours'] = float(np.mean([g['prompt_matches_ours'] for g in G]))
    res['trained'] = dict(groups_trained=sum(g['trained'] for g in G), groups=len(G))
    # per-batch trained fraction vs score
    tf = collections.defaultdict(list)
    for g in G: tf[g['uid']].append(g['trained'])
    tfa = np.array([np.mean(tf[u]) for u in sall]); res['trained']['spearman_score_vs_trained_frac'] = spearman(tfa, sv)
    # label-corruption control: flip correctness of a random fraction f of rollouts per batch, re-score (all_subjects, both halves)
    K = len(chunks); subj = sorted({c['subject'] for c in chunks if c['subject']})
    allk = [k for k, c in enumerate(chunks) if c['kind'] == 'subject']
    rng = np.random.default_rng(7); byu = collections.defaultdict(list)
    for g in G: byu[g['uid']].append(g)
    def sc(gs, f):
        N = sum(len(g['lens']) for g in gs); s = np.zeros(K)
        for g in gs:
            cls = list(g['claimed_cls'])
            for j in range(len(cls)):
                if rng.random() < f: cls[j] = 'negative' if cls[j] == 'positive' else 'positive'
            r = np.array(rew(cls, g['lens'])); s += (r - r.mean()) @ np.array(g['dots'])
        return s[allk].mean() / (8192 * N)
    nullsd = {r['uid']: math.sqrt(r['all_subjects_0_nullsd'] ** 2 + r['all_subjects_1_nullsd'] ** 2) / 2 for r in rows}
    base_s = {u: sc(gs, 0.0) for u, gs in byu.items()}
    corr = {}
    for f in (0.1, 0.25, 0.5, 1.0):
        d = np.array([(sc(gs, f) - base_s[u]) / nullsd[u] for u, gs in byu.items()])
        corr[str(f)] = dict(mean_shift_in_null_sd=float(d.mean()), median_abs_shift=float(np.median(np.abs(d))))
    res['label_corruption'] = corr
    out[E] = dict(summary=res, rows=rows)
    scores[E] = {r['uid']: dict(all=(r['all_subjects_0'] + r['all_subjects_1']) / 2, claimed=(r['claimed_0'] + r['claimed_1']) / 2,
                                pool=(r['pool_0'] + r['pool_1']) / 2) for r in rows}
    s = res; print('==', E, s['meta']['batches'], 'batches', s['meta']['chunks'], 'chunks', 'label', s['label_audit']['frac_wrong'], 'prompt', s['prompt_matches_ours'])
    for k in ('claimed', 'pool', 'wrong_claim', 'all_subjects'):
        m = s[k]; print(f"{k:13s} SH r {m['split_half_pearson']:+.2f} rho {m['split_half_spearman']:+.2f} SB {m['spearman_brown_full'] if m['spearman_brown_full'] is None else round(m['spearman_brown_full'],2)} z>2 {m['n_pos_z_gt2']} z<-2 {m['n_neg_z_lt_m2']} |z|med {m['snr_median_abs_z']:.2f} sep {m['frac_pairs_separable']:.3f} mean {m['mean']:+.2e} sdB {m['sd_across_batches']:.2e} nullsd {m['median_null_sd']:.2e} refnoise {m['ref_noise_sd_per_half']:.2e}", flush=True)
    for k in ('placebo', 'hottest_attack', 'wrong_claim_vs_claimed', 'rank_holdout', 'reference_overlap', 'trained', 'label_corruption'):
        print(k, json.dumps(s[k]))
# cross-epoch UID stability
cross = {}
Es = sorted(scores)
for i in range(len(Es)):
    for j in range(i + 1, len(Es)):
        a, b = Es[i], Es[j]; us = sorted(set(scores[a]) & set(scores[b]))
        cross[f'{a}-{b}'] = dict(uids=len(us), **{k: spearman(np.array([scores[a][u][k] for u in us]), np.array([scores[b][u][k] for u in us])) for k in ('all', 'claimed', 'pool')})
# UID-mean over all epochs: split epochs odd/even -> reliability of a multi-epoch average
if len(Es) >= 4:
    ev, od = Es[0::2], Es[1::2]; us = sorted(set.intersection(*[set(scores[e]) for e in Es]))
    z = lambda e, u: scores[e][u]['all'] / np.std([scores[e][v]['all'] for v in scores[e]])
    A = np.array([np.mean([z(e, u) for e in ev]) for u in us]); B = np.array([np.mean([z(e, u) for e in od]) for u in us])
    cross['multi_epoch_even_vs_odd'] = dict(uids=len(us), even=ev, odd=od, spearman=spearman(A, B), pearson=float(np.corrcoef(A, B)[0, 1]))
print('cross-epoch', json.dumps(cross, indent=0))
# regenerated vs real (same epoch, same reference, same uids)
reg = json.load(open(HERE / 'data/claims_results.json')); rvr = {}
for E in ('62', '64'):
    if E not in scores or E not in reg: continue
    rr = {r['uid']: r for r in reg[E]['rows']}; us = sorted(set(rr) & set(scores[E]))
    for k, name in (('all', 'all_subjects'), ('claimed', 'claimed'), ('pool', 'pool')):
        a = np.array([(rr[u][f'{name}_0'] + rr[u][f'{name}_1']) / 2 for u in us]); b = np.array([scores[E][u][k] for u in us])
        rvr.setdefault(E, {})[k] = dict(uids=len(us), spearman=spearman(a, b), sd_regen=float(a.std()), sd_real=float(b.std()),
                                       mean_regen=float(a.mean()), mean_real=float(b.mean()))
print('regen-vs-real', json.dumps(rvr, indent=0))
json.dump(dict(epochs=out, cross_epoch=cross, regen_vs_real=rvr), open(HERE / 'data' / 'claims_real_results.json', 'w'), default=str)
