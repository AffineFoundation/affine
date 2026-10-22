#!/usr/bin/env python3
"""Alternative scoring rules on the SAME stored gradient-dots (CPU only).
Per batch b: chunk vector s_b[k] = sum_i A_i d_i[k] / (8192 N)  (GRPO direction), and
             sft_b[k] = sum_{i claimed correct} d_i[k] / (8192 N)  (plain log-likelihood gradient of the batch's correct completions).
Rules: raw (all-subject chunks, both halves = current), z_null (raw / shuffled-label SD), chunk_cos (mean/rms over chunks),
       subj_mean (equal weight per subject), subj_max (max over subjects), one_draw (half 0 only), sft (loss-agnostic)."""
import glob, gzip, json, sys, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import grader
from plant_dots import key
rew = lambda c, L: grader.overlong_reward(1.0 if c == 'positive' else 0.0, L)
def spearman(a, b):
    a, b = np.asarray(a), np.asarray(b); return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))
RR = json.load(open(HERE / 'data/claims_real_results.json')); PR = json.load(open(HERE / 'data/planted_results.json'))
RULES = ['raw', 'z_null', 'chunk_cos', 'subj_mean', 'subj_max', 'one_draw', 'sft']
def rules(s, sft, chunks, nullsd):
    sub = [k for k, c in enumerate(chunks) if c['kind'] == 'subject']; h0 = [k for k in sub if chunks[k]['half'] == 0]; h1 = [k for k in sub if chunks[k]['half'] == 1]
    raw = (s[h0].mean() + s[h1].mean()) / 2
    bys = collections.defaultdict(list)
    for k in sub: bys[chunks[k]['subject']].append(k)
    per = np.array([s[v].mean() for v in bys.values()])
    return dict(raw=raw, z_null=raw / nullsd if nullsd else 0.0, chunk_cos=s[sub].mean() / (np.sqrt((s[sub] ** 2).mean()) + 1e-30),
                subj_mean=per.mean(), subj_max=per.max(), one_draw=s[h0].mean(), sft=(sft[h0].mean() + sft[h1].mean()) / 2)
def vec(groups):   # groups: list of (rewards, claimed_correct_mask, dots[8,K])
    N = sum(len(r) for r, _, _ in groups); K = groups[0][2].shape[1]; s = np.zeros(K); f = np.zeros(K)
    for r, c, d in groups: s += (r - r.mean()) @ d; f += c.astype(float) @ d
    return s / (8192 * N), f / (8192 * N)
real_scores = {}; out = {}
for E in ['62', '63', '64', '65']:
    D = [json.load(open(f)) for f in sorted(glob.glob(str(HERE / 'data' / f'e{E}' / 'dots-real-*.json')))]; chunks = D[0]['chunks']
    byu = collections.defaultdict(list)
    for d in D:
        for g in d['groups']:
            byu[g['uid']].append((np.array([rew(c, L) for c, L in zip(g['claimed_cls'], g['lens'])]), np.array([c == 'positive' for c in g['claimed_cls']]), np.array(g['dots'])))
    nsd = {r['uid']: np.sqrt(r['all_subjects_0_nullsd'] ** 2 + r['all_subjects_1_nullsd'] ** 2) / 2 for r in RR['epochs'][E]['rows']}
    real_scores[E] = {u: rules(*vec(gs), chunks, nsd.get(u)) for u, gs in byu.items()}
    if E not in PR: continue
    # planted batches
    P = HERE / 'data' / 'planted' / f'e{E}'; spec = json.load(open(P / 'spec.json'))
    RD = {(g['miner'], g['slot']): g['dots'] for d in D for g in d['groups']}
    RG = {(g['uid'], g['index']): g['dots'] for f in glob.glob(str(HERE / 'data' / f'e{E}' / 'dots-[0-9]*.json')) for g in json.load(open(f))['groups'] if g['valid']}
    cache = {}
    for f in glob.glob(str(P / 'cache-*.json')): cache.update(json.load(open(f)))
    m = json.load(open(P / 'new.json')); z = np.load(P / 'new.npz'); o = z['offsets']; tk = z['tokens']
    NEW = [cache.get(key(x['index'], tk[o[n]:o[n + 1]])) for n, x in enumerate(m['meta'])]
    def dots(r):
        if r['src'] == 'real': return RD[tuple(r['key'][:2])][r['key'][2]]
        if r['src'] == 'regen': return RG[(r['key'][0], r['key'][1])][r['key'][2]]
        return NEW[r['key']]
    pn = {r['id']: r['null_sd'] for r in PR[E]['rows']}
    pl = collections.defaultdict(list)
    for b in spec['batches']:
        gs = [(np.array([r['reward'] for r in g['rollouts']]), np.array([r["reward"] > 0.25 for r in g['rollouts']]), np.array([dots(r) for r in g['rollouts']])) for g in b['groups']]
        pl[b['type']].append(rules(*vec(gs), chunks, pn.get(b['id'])))
    res = {}
    for rule in RULES:
        honest = [x[rule] for x in pl['1_honest']] + [x[rule] for x in real_scores[E].values()]
        good = [x[rule] for t in pl if t[0] in '78' for x in pl[t]]; bad = [x[rule] for t in pl if t[0] in '234' for x in pl[t]]
        tm = {t: float(np.mean([x[rule] for x in pl[t]])) for t in pl}
        res[rule] = dict(auc_good_vs_honest=auc(good, honest), auc_honest_vs_bad=auc(honest, bad), auc_beyond_policy_vs_honest=auc([x[rule] for x in pl.get('8_beyond_policy_bo48', [])], honest) if pl.get('8_beyond_policy_bo48') else None,
                         auc_good_balanced_vs_honest=auc([x[rule] for x in pl['7_good_balanced']], honest), ranking=sorted(tm, key=lambda t: -tm[t]), type_means=tm)
    out[E] = res
# cross-epoch UID consistency on real miners (all 6 pairs of steps 62-65)
cross = {}
Es = sorted(real_scores)
for rule in RULES:
    v = []
    for i in range(len(Es)):
        for j in range(i + 1, len(Es)):
            us = sorted(set(real_scores[Es[i]]) & set(real_scores[Es[j]]))
            v.append(spearman([real_scores[Es[i]][u][rule] for u in us], [real_scores[Es[j]][u][rule] for u in us]))
    cross[rule] = dict(pairs=v, mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)))
# type-ranking consistency across the two planted epochs
tr = {}
if len(out) == 2:
    a, b = sorted(out)
    for rule in RULES:
        ts = sorted(set(out[a][rule]['type_means']) & set(out[b][rule]['type_means']))
        tr[rule] = spearman([out[a][rule]['type_means'][t] for t in ts], [out[b][rule]['type_means'][t] for t in ts])
json.dump(dict(planted=out, real_cross_epoch=cross, type_rank_cross_epoch=tr), open(HERE / 'data' / 'planted_rules.json', 'w'))
print('| rule | AUC good vs honest (62/64) | AUC beyond-policy vs honest | AUC good-balanced vs honest | AUC honest vs bad | type-rank Spearman 62↔64 | real-miner cross-epoch UID ρ mean [min,max] |')
for rule in RULES:
    g = '/'.join(f"{out[E][rule]['auc_good_vs_honest']:.2f}" for E in sorted(out)); bp = '/'.join(f"{out[E][rule]['auc_beyond_policy_vs_honest']:.2f}" for E in sorted(out))
    gb = '/'.join(f"{out[E][rule]['auc_good_balanced_vs_honest']:.2f}" for E in sorted(out)); hb = '/'.join(f"{out[E][rule]['auc_honest_vs_bad']:.2f}" for E in sorted(out))
    c = cross[rule]; print(f"| {rule} | {g} | {bp} | {gb} | {hb} | {tr.get(rule, float('nan')):+.2f} | {c['mean']:+.3f} [{c['min']:+.2f},{c['max']:+.2f}] |")
for E in sorted(out):
    for rule in RULES: print(E, rule, [t.split('_')[0] for t in out[E][rule]['ranking']])
