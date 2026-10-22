#!/usr/bin/env python3
"""Build data/adv/e<E>/scratch-spec.json: 3-group batches for the Goodhart scratch-step check."""
import glob, gzip, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import grader
E = sys.argv[1]; extra = sys.argv[2].split(',') if len(sys.argv) > 2 else []
rng = np.random.default_rng(int(E) * 31 + 5)
D = [json.load(open(f)) for f in sorted(glob.glob(str(HERE / 'data' / f'e{E}' / 'dots-real-*.json')))]; ch = D[0]['chunks']
h0 = [k for k, c in enumerate(ch) if c['kind'] == 'subject' and c['half'] == 0]; reft = {i for c in ch for i in c['tasks']}
R = {(r['miner'], r['slot']): r for r in (json.loads(l) for l in gzip.open(HERE / 'data/real' / f'real-{E}.jsonl.gz', 'rt'))}
G = []
for d in D:
    for g in d['groups']:
        r = R[(g['miner'], g['slot'])]
        if r['index'] in reft: continue
        rw = [grader.overlong_reward(1.0 if c == 'positive' else 0.0, L) for c, L in zip(g['claimed_cls'], g['lens'])]
        if np.std(rw) == 0: continue
        A = np.array(rw) - np.mean(rw); s0 = float((A @ np.array(g['dots']))[h0].mean())
        G.append(dict(index=r['index'], rollouts=[dict(src='real', key=[g['miner'], g['slot'], j], reward=float(x)) for j, x in enumerate(rw)], s0=s0, pas=float(np.mean([c == 'positive' for c in g['claimed_cls']]))))
batches = []
def add(t, n, groups): batches.append(dict(id=f'{t}-{n}', type=t, groups=[{k: v for k, v in g.items() if k in ('index', 'rollouts')} for g in groups]))
for n in range(3): add('honest', n, [G[j] for j in rng.choice(len(G), 3, replace=False)])
for n in range(3):
    c = [G[j] for j in rng.choice(len(G), 96, replace=False)]; add('cherry_h0_x32', n, sorted(c, key=lambda g: -g['s0'])[:3])
for n in range(2):
    c = [G[j] for j in rng.choice(len(G), 96, replace=False)]; add('length_x32', n, sorted(c, key=lambda g: -np.mean([R[tuple(r['key'][:2])]['rollouts'][r['key'][2]].get('len', len(R[tuple(r['key'][:2])]['rollouts'][r['key'][2]]['out'])) for r in g['rollouts']]))[:3])
easy = [g for g in G if g['pas'] >= 0.75]
for n in range(2): add('easy', n, [easy[j] for j in rng.choice(len(easy), 3, replace=False)])
spec = json.load(open(HERE / 'data' / 'planted' / f'e{E}' / 'spec.json'))
bp = [b for b in spec['batches'] if b['type'].startswith('8')]
for n in range(3): add('beyond_policy', n, bp[n]['groups'][:3])
gb = [b for b in spec['batches'] if b['type'].startswith('4')]
for n in range(2):
    gs = []
    for g in gb[n]['groups'][:3]:
        gs.append(dict(index=g['index'], rollouts=[dict(r, reward=0.0) if r['src'] == 'new' else r for r in g['rollouts']]))
    add('garbage_as_wrong', n, gs)
# extra adv sets (distill/copies) appended as groups of: <=2 correct from adv set + 6 wrong base reference samples
import grad_real, collections
if extra:
    base, meta = grad_real.load_ref(E); byref = collections.defaultdict(list)
    for x in meta:
        if x['kind'] == 'ref': byref[x['index']].append(x)
    refj = {i: [j for j, x in enumerate(xs) if x['outcome'] != 'positive'][:6] for i, xs in byref.items()}
    refl = {i: xs for i, xs in byref.items()}
    for nm in extra:
        m = json.load(open(HERE / 'data' / 'adv' / f'e{E}' / f'{nm.split(":")[0]}.json'))['meta']; var = nm.split(':')[1] if ':' in nm else None
        by = collections.defaultdict(list)
        for j, x in enumerate(m):
            if x['outcome'] == 'positive' and (var is None or x['variant'] == var): by[x['index']].append(j)
        groups = []
        for i, js in sorted(by.items()):
            if len(refj.get(i, [])) < 6: continue
            ro = [dict(src='adv', key=[nm.split(':')[0], j], reward=float(grader.overlong_reward(1.0, m[j]['len']))) for j in js[:2]]
            ro += [dict(src='ref', key=[i, j], reward=float(grader.overlong_reward(0.0, refl[i][j]['len']))) for j in refj[i]]
            groups.append(dict(index=i, rollouts=ro))
        rng.shuffle(groups)
        for n in range(min(2, len(groups) // 3)): add(f'adv_{nm.replace(":", "_")}', n, groups[n * 3:(n + 1) * 3])
out = HERE / 'data' / 'adv' / f'e{E}'; out.mkdir(parents=True, exist_ok=True)
json.dump(dict(epoch=E, batches=batches), open(out / 'scratch-spec.json', 'w'))
print('spec', E, [b['id'] for b in batches])
