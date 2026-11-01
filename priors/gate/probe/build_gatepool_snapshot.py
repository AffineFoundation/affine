#!/usr/bin/env python3
"""SN120 prior-v1 gate pool -> probe task snapshot (RA/20261101-baseline-switch, probe kit).
usage: build_gatepool_snapshot.py MATH_SNAPSHOT GATE_POOL_JSONL OUT_SNAPSHOT
Probe index i: i < 7496 = affine_math snapshot entry i, byte-for-byte the live one; 7496 + j = dapo:j of the public gate pool, written
as a MathTask cloned from math entry 0 (same task_class/task_config/system_prompt/limits) with problem/prompt/answer/idx/name replaced.
The 4 affine_math tasks dropped from the public pool stay in the snapshot so indices do not move (they are never drawn)."""
import copy, hashlib, json, os, sys
MATH_N = 7496
src, pool, out = sys.argv[1:4]
math = json.load(open(src)); assert isinstance(math, list) and len(math) == MATH_N, 'math snapshot must have %d tasks' % MATH_N
rows = [json.loads(l) for l in open(pool)]
by_id = {r['id']: r for r in rows}
for r in rows:   # the public pool must agree with the live snapshot on every math task it keeps
    if r['env'] == 'affine_math':
        d = math[int(r['id'].split(':')[1])]['data']; assert d['problem'] == r['problem'] and str(d['answer']) == str(r['answer']), r['id']
dapo = sorted((r for r in rows if r['env'] == 'dapo'), key=lambda r: int(r['id'].split(':')[1]))
assert [int(r['id'].split(':')[1]) for r in dapo] == list(range(len(dapo))), 'dapo ids must be 0..n-1'
tmpl = math[0]; snap = list(math)
for j, r in enumerate(dapo):
    t = copy.deepcopy(tmpl); d = t['data']
    d.update(problem=r['problem'], prompt=r['problem'], answer=str(r['answer']), idx=MATH_N + j,
             name='dapo-' + hashlib.sha256(r['problem'].encode()).hexdigest()[:12], level='DAPO', subject='DAPO')
    snap.append(t)
tmp = out + '.tmp'; open(tmp, 'w').write(json.dumps(snap, ensure_ascii=False)); os.replace(tmp, out)
h = hashlib.sha256(open(out, 'rb').read()).hexdigest()
print(json.dumps(dict(out=out, tasks=len(snap), math=MATH_N, dapo=len(dapo), sha256=h,
                      math_snapshot_sha256=hashlib.sha256(open(src, 'rb').read()).hexdigest())))
