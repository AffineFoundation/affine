#!/usr/bin/env python3
"""Pool all sampler records per (before, after) checkpoint pair.

Each record is an independent paired estimate (its own random K). Pooling is done at the
task level: all per-task paired differences from all instances are concatenated (tasks are
iid uniform draws from the pool in every instance), giving pooled delta and SE. Also prints
an inverse-variance combination of per-record deltas as a cross-check.
usage: aggregate.py results/ [--metric correctness|reward] [--crn any|on|off] [--json]
(--crn on keeps records whose before/after seeds were shared: common_random_numbers or legacy common_random_seeds.)
"""
import argparse, glob, json, math, os
from collections import defaultdict

ap = argparse.ArgumentParser(); ap.add_argument('results', nargs='+')
ap.add_argument('--metric', choices=['correctness', 'reward'], default='correctness')
ap.add_argument('--crn', choices=['any', 'on', 'off'], default='any')
ap.add_argument('--json', action='store_true'); a = ap.parse_args()
def is_crn(rec):
    g = rec.get('generation', {}); return bool(g.get('common_random_numbers', g.get('common_random_seeds', False)))
key = ('before_correct', 'after_correct') if a.metric == 'correctness' else ('before_reward', 'after_reward')
pairs = defaultdict(list)
for r in a.results:
    files = glob.glob(os.path.join(r, '*.jsonl')) if os.path.isdir(r) else [r]
    for f in files:
        for line in open(f):
            line = line.strip()
            if line:
                rec = json.loads(line)
                if a.crn != 'any' and is_crn(rec) != (a.crn == 'on'):
                    continue
                pairs[(rec['before']['checkpoint_id'], rec['after']['checkpoint_id'])].append(rec)
out = []
for (b, c), recs in sorted(pairs.items(), key=lambda kv: kv[1][0]['after'].get('optimizer_step') or 0):
    d, bb, aa = [], [], []
    for rec in recs:
        for t in rec['tasks']:
            if t[key[0]] is not None and t[key[1]] is not None:
                d.append(t[key[1]] - t[key[0]]); bb.append(t[key[0]]); aa.append(t[key[1]])
    n = len(d)
    if n < 2:
        continue
    mu = sum(d) / n; var = lambda v: sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)
    se = math.sqrt(var(d) / n); se_u = math.sqrt(var(bb) / n + var(aa) / n)
    w = [(rec[a.metric]['delta'], rec[a.metric]['se_paired']) for rec in recs if rec[a.metric].get('se_paired')]
    ivw = (sum(x / s ** 2 for x, s in w) / sum(1 / s ** 2 for x, s in w), math.sqrt(1 / sum(1 / s ** 2 for x, s in w))) if w else (None, None)
    row = dict(step=recs[0]['after'].get('optimizer_step'), before=b[:8], after=c[:8], records=len(recs),
               instances=len({r['sampler_id'] for r in recs}), tasks=n, before_mean=sum(bb) / n, after_mean=sum(aa) / n,
               delta=mu, se_paired=se, se_unpaired=se_u, z=mu / se if se else None, ivw_delta=ivw[0], ivw_se=ivw[1])
    out.append(row)
if a.json:
    print(json.dumps(out, indent=1))
else:
    print(f"{'step':>4} {'before':>8} -> {'after':8} {'rec':>3} {'tasks':>5} {'before':>6} {'after':>6} {'delta':>8} {'±SEpair':>8} {'SEunpair':>8} {'z':>6}")
    for r in out:
        print(f"{r['step']:>4} {r['before']:>8} -> {r['after']:8} {r['records']:>3} {r['tasks']:>5} {r['before_mean']:6.3f} {r['after_mean']:6.3f} "
              f"{r['delta']:+8.4f} {r['se_paired']:8.4f} {r['se_unpaired']:8.4f} {r['z']:+6.2f}")
