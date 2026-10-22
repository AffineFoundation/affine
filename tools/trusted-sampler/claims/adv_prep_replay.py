#!/usr/bin/env python3
"""Replay set: epoch-SRC beyond-policy (bo16) groups (teacher correct j<16, first<=2, + 6 wrong SRC-base reference samples), stored as
data/adv/e<DST>/replay<SRC>.npz/json for scoring at base(DST)/V(DST). Also records whether each task is in DST's reference chunks."""
import json, sys, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import grad_real, grad_claims as gc, grader
SRC, DST = sys.argv[1], sys.argv[2]
base, meta = grad_real.load_ref(SRC); P = HERE / 'data' / 'planted' / f'e{SRC}'
teach = json.load(open(P / 'teacher.json')); tz = np.load(P / 'teacher.npz'); to = tz['offsets']; tt = tz['tokens']
byref = collections.defaultdict(list)
for x in meta:
    if x['kind'] == 'ref': byref[x['index']].append(x)
cor = collections.defaultdict(list)
for k, c in enumerate(teach['correct']):
    if c['j'] < 16: cor[c['index']].append((c['j'], k))
_, dmeta = grad_real.load_ref(DST); dch, _, _ = gc.build_chunks(dmeta, DST); dref = {i for c in dch for i in c['tasks']}
toks, offs, rec = [], [0], []
for i, cs in sorted(cor.items()):
    cs = sorted(cs)[:2]; wr = [x for x in byref[i] if x['outcome'] != 'positive'][:6]
    for j, k in cs:
        y = tt[to[k]:to[k + 1]]; toks.append(np.asarray(y, np.int32)); offs.append(offs[-1] + len(y)); rec.append(dict(index=i, role='correct', reward=grader.overlong_reward(1.0, len(y)), len=len(y), in_dst_ref=i in dref))
    for x in wr:
        y = np.asarray(x['tokens'], np.int32); toks.append(y); offs.append(offs[-1] + len(y)); rec.append(dict(index=i, role='wrong', reward=grader.overlong_reward(0.0, x['len']), len=len(y), in_dst_ref=i in dref))
out = HERE / 'data' / 'adv' / f'e{DST}'; out.mkdir(parents=True, exist_ok=True)
np.savez(out / f'replay{SRC}.npz', tokens=np.concatenate(toks), offsets=np.asarray(offs, np.int64))
json.dump(dict(src=SRC, dst=DST, meta=rec), open(out / f'replay{SRC}.json', 'w'))
print('replay', SRC, '->', DST, 'groups', len(cor), 'seqs', len(rec), 'tasks in dst ref', len({r['index'] for r in rec if r['in_dst_ref']}))
