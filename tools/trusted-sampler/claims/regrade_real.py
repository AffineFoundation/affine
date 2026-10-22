#!/usr/bin/env python3
"""CPU: re-grade every real miner rollout with our local verifier replica (grader.py) and compare to the miner's claim.
Output claims/data/real/regrade-<E>.json : per doc (uid, slot, index) our classification per rollout."""
import gzip, json, sys, glob, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT))
import grader
from transformers import AutoTokenizer
T = json.load(open(ROOT / 'data/original-math7496.tasks.json'))
tok = AutoTokenizer.from_pretrained(glob.glob('/root/ckpt-cache/c8a6d342*')[0]); special = set(tok.all_special_ids); vocab = 248077
gp = grader.pool(48)
for E in sys.argv[1].split(','):
    R = [json.loads(l) for l in gzip.open(HERE / 'data/real' / f'real-{E}.jsonl.gz', 'rt')]
    flat = [(n, j) for n, r in enumerate(R) for j in range(len(r['rollouts']))]
    pr = [grader.project(tok, R[n]['rollouts'][j]['out'], special, vocab) for n, j in flat]
    kinds = list(gp.map(grader.classify, [p[0] for p in pr], [T[R[n]['index']]['data']['answer'] for n, _ in flat], chunksize=16))
    out = [dict(uid=r['uid'], miner=r['miner'], slot=r['slot'], index=r['index'], ours=[None] * len(r['rollouts']), reason=[None] * len(r['rollouts'])) for r in R]
    for (n, j), (k, c), p in zip(flat, kinds, pr): out[n]['ours'][j] = k; out[n]['reason'][j] = p[1]
    conf = collections.Counter((R[n]['rollouts'][j]['cls'], out[n]['ours'][j]) for n, j in flat)
    json.dump(dict(epoch=E, docs=out, confusion={f'{a}->{b}': v for (a, b), v in conf.items()}), open(HERE / 'data/real' / f'regrade-{E}.json', 'w'))
    print(E, dict(conf), flush=True)
