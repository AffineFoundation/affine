#!/usr/bin/env python3
"""CRN vs no-CRN paired SE at equal K, per pair. usage: crn_compare.py results/ [pairs...]"""
import json, glob, math, sys, statistics as st
recs = [json.loads(l) for f in glob.glob(sys.argv[1].rstrip('/') + '/*.jsonl') for l in open(f)]
def crn(r):
    g = r['generation']; return g.get('common_random_numbers', g.get('common_random_seeds'))
out = {}
for r in recs:
    key = (r['before']['checkpoint_id'][:8], r['after']['checkpoint_id'][:8], r['after']['optimizer_step'], r['K'], r['samples_per_task'], bool(crn(r)))
    out.setdefault(key, []).append(r)
rows = []
for (b, a, step, K, m, c), rs in sorted(out.items()):
    if K != 128: continue
    d = [t['after_correct'] - t['before_correct'] for r in rs for t in r['tasks'] if t['before_n'] and t['after_n']]
    bb = [t['before_correct'] for r in rs for t in r['tasks'] if t['before_n'] and t['after_n']]
    aa = [t['after_correct'] for r in rs for t in r['tasks'] if t['before_n'] and t['after_n']]
    sd_d = st.stdev(d); rho = st.correlation(bb, aa)
    per = [r['se'] for r in rs]
    rows.append(dict(pair=f'{b}->{a}', step=step, crn=c, records=len(rs), tasks=len(d), delta=st.mean(d),
                     task_sd_delta=sd_d, se_per_K128_record=sd_d / math.sqrt(128), mean_record_se=st.mean(per),
                     pooled_se=sd_d / math.sqrt(len(d)), corr_before_after=rho,
                     unpaired_se_K128=math.sqrt((st.pvariance(bb) + st.pvariance(aa)) / 128)))
for r in rows:
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}))
