#!/usr/bin/env python3
"""CountSketch (gnorm.py machinery, same seed) for the beyond-policy bo16 groups (2 teacher-correct + 6 reference-wrong per task) and the adversarial
sets in data/adv/e<E>/ (copies/distill positives, replay62 all), so group gradient norms exist for attack AND legit groups (cosine-normalised score test)."""
import collections, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import gnorm, grad_real
def items(E):
    T = []; P = HERE / 'data' / 'planted' / f'e{E}'; teach = json.load(open(P / 'teacher.json')); tz = np.load(P / 'teacher.npz'); o = tz['offsets']; t = tz['tokens']
    _, meta = grad_real.load_ref(E); byref = collections.defaultdict(list)
    for x in meta:
        if x['kind'] == 'ref': byref[x['index']].append(x)
    cor = collections.defaultdict(list)
    for k, c in enumerate(teach['correct']):
        if c['j'] < 16: cor[c['index']].append((c['j'], k))
    for i, cs in cor.items():
        for j, k in sorted(cs)[:2]: T.append((i, t[o[k]:o[k + 1]]))
    for i in byref:   # wrong reference samples (first 6 per task) for all tasks: used by BP, copies, distill groups
        for x in [x for x in byref[i] if x['outcome'] != 'positive'][:6]: T.append((i, np.asarray(x['tokens'], dtype=np.int32)))
    A = HERE / 'data' / 'adv' / f'e{E}'
    for nm in ('copies', 'distill', 'replay62'):
        if not (A / f'{nm}.json').exists(): continue
        m = json.load(open(A / f'{nm}.json'))['meta']; z = np.load(A / f'{nm}.npz'); oo = z['offsets']; tk = z['tokens']
        for n, x in enumerate(m):
            if nm == 'replay62' or x['outcome'] == 'positive': T.append((x['index'], tk[oo[n]:oo[n + 1]]))
    return T
gnorm.items = items
if __name__ == '__main__':
    gnorm.main()
