import json, os, sys, numpy as np
import compare as CP
R = json.load(open(CP.OUT / (os.environ.get('OUTNAME', 'COMPARE') + '.private.json')))
# mix weights over simulated targets from the live gain histogram of the 59 prior-v1 turns
g = [r['gain'] for r in json.load(open(CP.OUT.parent / 'REPLAY-replay.private.json'))['rows']] if 0 else []
import glob
for f in glob.glob(str(CP.OUT.parent / 'REPLAY*replay*.json')):
    J = json.load(open(f)); rows = J if isinstance(J, list) else J.get('replay', [])
    g += [r['gain'] for r in rows if isinstance(r, dict) and 'gain' in r]
g = np.array(g); T = [-0.01, -0.005, 0.0, 0.003, 0.005, 0.01]; edges = [-9, -0.0075, -0.0025, 0.0015, 0.004, 0.0075, 9]
W = np.histogram(g, bins=edges)[0] / len(g); print('live gains n=%d mix weights' % len(g), dict(zip(T, np.round(W, 2))))
for tag, run in R['runs'].items():
    print('\n==', tag, 's_eps', R['s_eps'], 'alpha_today(sim)=%.3f' % run['alpha_today'], 'sdA=%.5f sdB=%s' % (run['sd_gain_A'], run['sd_gain_B']))
    print('%-34s %6s %6s %6s %6s %6s | tasks@ -1 -.5 0 +.3 +.5 +1 | mix' % ('design|rule', 'c', 'FG0', '+.3', '+.5', '+1'))
    keep = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    for k, r in sorted(run['ops'].items()):
        if keep and not any(x == k.split('|', 1)[1] for x in keep): continue
        mix = sum(w * r['tasks_%g' % t] for w, t in zip(W, T))
        print('%-34s %6.3f %6.3f %6.3f %6.3f %6.3f | %s | %4.0f' % (k, r['c'], r['green_0'], r['green_0.003'], r['green_0.005'], r['green_0.01'], ' '.join('%4.0f' % r['tasks_%g' % t] for t in T), mix))
