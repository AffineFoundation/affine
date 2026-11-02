#!/usr/bin/env python3
"""Paired replay analysis: per GREEN file, holdout-subset pass delta of the step_mult-m candidate vs its head (m=0); scaled to the
full 4,976-task holdout (tasks outside the subset have historical pass rate <0.03 or >0.97 and are treated as unchanged)."""
import json, math, os, sys
R = '/data/smult'; FULL = 4976
turns = json.load(open(R + '/turns.json'))
def key(t, m): return ('ck-' + t['H'][:12]) if m == 0 else ('ck-' + t['C'][:12]) if m == 1 else 'e%d-m%d' % (t['epoch'], m)
def grades(k):
    p = '%s/evals/%s/result.json.grades.jsonl' % (R, k)
    if not os.path.exists(R + '/evals/%s/result.json' % k): return None
    g = {}
    for l in open(p):
        d = json.loads(l)
        if d.get('set') == 'holdout' and d.get('n'): g[d['id']] = d['c'] / d['n']
    return g
MS = [1, 2, 4, 8]; rows = []; agg = {m: [] for m in MS}
for t in turns:
    g0 = grades(key(t, 0)); r = dict(epoch=t['epoch'])
    if g0 is None: rows.append(r); continue
    r['head_pass_subset'] = sum(g0.values()) / len(g0)
    for m in MS:
        gm = grades(key(t, m))
        if gm is None: continue
        ids = sorted(set(g0) & set(gm)); n = len(ids); d = [gm[i] - g0[i] for i in ids]; mu = sum(d) / n
        var = sum((x - mu) ** 2 for x in d) / (n - 1); se = math.sqrt(var / n); f = n / FULL
        r['m%d' % m] = dict(delta_pp_full=100 * mu * f, se_pp_full=100 * se * f, n=n, pass_subset=sum(gm[i] for i in ids) / n)
        agg[m].append((mu * f, (se * f) ** 2))
    rows.append(r)
summary = {}
for m in MS:
    a = agg[m]
    if not a: continue
    mu = sum(x for x, _ in a) / len(a); se = math.sqrt(sum(v for _, v in a)) / len(a)
    summary['m%d' % m] = dict(files=len(a), mean_delta_pp_full=100 * mu, se_pp=100 * se, z=mu / se if se else None,
                              wins=sum(1 for x, _ in a if x > 0))
out = dict(rows=rows, summary=summary)
json.dump(out, open(R + '/analysis.json', 'w'), indent=1)
print('%-6s %8s' % ('epoch', 'head%') + ''.join('%18s' % ('m=%d dpp(se)' % m) for m in MS))
for r in rows:
    print('%-6s %8s' % (r['epoch'], '%.2f' % (100 * r['head_pass_subset']) if 'head_pass_subset' in r else '-') +
          ''.join('%18s' % ('%+.2f (%.2f)' % (r['m%d' % m]['delta_pp_full'], r['m%d' % m]['se_pp_full']) if 'm%d' % m in r else '-') for m in MS))
for m, s in summary.items(): print(m, json.dumps(s))
