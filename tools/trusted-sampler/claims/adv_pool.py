#!/usr/bin/env python3
"""affine_math task pool + overlap analysis (CPU). Aggregates only."""
import glob, gzip, json, sys, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import grader, grad_real
T = json.load(open(HERE.parent / 'data/original-math7496.tasks.json')); idx = json.load(open(HERE.parent / 'data/index.json'))
pool = idx['pool']['indices']; P = set(pool); rng = np.random.default_rng(7)
out = dict(snapshot=len(T), pool=len(pool), distinct_problems=len({t['data']['problem'] for t in T}), distinct_answers=len({t['data']['answer'] for t in T}),
           pool_subjects=dict(collections.Counter(T[i]['data']['subject'] for i in pool)), nonpool_subjects=dict(collections.Counter(T[i]['data']['subject'] for i in range(len(T)) if i not in P)),
           pool_levels=dict(collections.Counter(T[i]['data']['level'] for i in pool)))
rew = lambda c, L: grader.overlong_reward(c, L)
E_OUT = {}; seen_all = collections.Counter()
for E in ('62', '64', '66'):
    R = [json.loads(l) for l in gzip.open(HERE / 'data/real' / f'real-{E}.jsonl.gz', 'rt')]
    tasks = collections.Counter(r['index'] for r in R); seen_all.update(set(tasks))
    base, meta = grad_real.load_ref(E); reft = collections.defaultdict(list)
    for x in meta:
        if x['kind'] == 'ref': reft[x['index']].append(x['outcome'] == 'positive')
    drawn = set(reft); band = {i for i, v in reft.items() if 2 <= sum(v) <= 6}
    pr = {i: np.mean(v) for i, v in reft.items()}
    # random K-task draws (K=1400 stratified like the live rule, approximated by uniform) -> fraction of submitted groups whose task is drawn / in band
    sub = np.array([r['index'] for r in R]); fr = []
    for _ in range(200):
        d = set(rng.choice(pool, 1400, replace=False).tolist()); fr.append(np.mean([i in d for i in sub]))
    # dots: in-sample groups, score with and without the chunks that contain the group's own task (leave-task-out)
    D = [json.load(open(f)) for f in sorted(glob.glob(str(HERE / 'data' / f'e{E}' / 'dots-real-*.json')))]; ch = D[0]['chunks']
    sk = [k for k, c in enumerate(ch) if c['kind'] == 'subject']; h0 = [k for k in sk if ch[k]['half'] == 0]; h1 = [k for k in sk if ch[k]['half'] == 1]
    w = np.zeros(len(ch)); w[h0] = .5 / len(h0); w[h1] = .5 / len(h1); chunk_tasks = [set(c['tasks']) for c in ch]
    s_full, s_lto, ins = [], [], []
    for d in D:
        for g in d['groups']:
            rws = np.array([rew(1.0 if c == 'positive' else 0.0, L) for c, L in zip(g['claimed_cls'], g['lens'])]); A = rws - rws.mean(); v = (A @ np.array(g['dots'])) / (8192 * len(rws))
            i = g['index']; m = np.array([i in t for t in chunk_tasks]); w2 = w.copy(); w2[m] = 0
            for hh in (h0, h1):
                hm = np.zeros(len(ch), bool); hm[hh] = True; tot = w[hm].sum(); rem = w2[hm].sum()
                if rem > 0: w2[hm] *= tot / rem
            s_full.append(v @ w); s_lto.append(v @ w2); ins.append(bool(m.any()))
    s_full, s_lto, ins = map(np.array, (s_full, s_lto, ins)); mu, sd = np.median(s_full[~ins]), s_full[~ins].std()
    z, zl = (s_full - mu) / sd, (s_lto - mu) / sd
    E_OUT[E] = dict(groups=len(R), distinct_tasks=len(tasks), frac_pool_submitted=len(tasks) / len(pool), max_groups_per_task=max(tasks.values()),
                    ref_drawn=len(drawn), ref_band=len(band), band_frac_of_drawn=len(band) / len(drawn),
                    frac_groups_task_in_random1400_draw=float(np.mean(fr)), frac_groups_task_in_actual_band=float(np.mean([i in band for i in sub])),
                    frac_band_tasks_submitted=float(np.mean([i in tasks for i in band])),
                    pool_minus_submitted=len(P - set(tasks)),
                    insample_groups=int(ins.sum()), insample_mean_z=float(z[ins].mean()), insample_mean_z_leave_task_out=float(zl[ins].mean()),
                    insample_frac_z_gt2=float((z[ins] > 2).mean()), insample_frac_z_gt2_lto=float((zl[ins] > 2).mean()), outsample_frac_z_gt2=float((z[~ins] > 2).mean()),
                    corr_full_lto_outsample=float(np.corrcoef(s_full[~ins], s_lto[~ins])[0, 1]))
out['epochs'] = E_OUT; out['distinct_tasks_submitted_3_epochs'] = len(seen_all); out['frac_pool_submitted_3_epochs'] = len(seen_all) / len(pool)
json.dump(out, open(HERE / 'data' / 'adv_pool.json', 'w'), indent=1); print(json.dumps(out, indent=1))
