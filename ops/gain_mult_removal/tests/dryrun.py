"""Dry run (no writes): staged (= pre-multiplier) k1_service.scores_doc vs live k1_service on the live inputs; staged vs live training_params."""
import importlib.util, json, sys, math
from pathlib import Path
R = Path('/home/const/subnet120-rewrite/state/root-audits/20261103-gain-mult-removal'); RA = R.parent
def load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
live = load('k1l', str(RA / '20261027-k1-per-uid/code/k1_service.py')); new = load('k1n', str(R / 'stage/k1_service.py'))
pre = load('k1p', str(RA / '20261102-gain-mult/backups/pre-20261102T224102Z/k1_service.py'))
chain = live.rj(live.ST / 'K1-CHAIN.json'); recs = live.verdicts(chain)
a, b, p = live.scores_doc(recs, chain, None), new.scores_doc(recs, chain, None), pre.scores_doc(recs, chain, None)
v = lambda d: [(m['uid'], round(m['share_now'], 12)) for m in d['miners']]
n = len(b['miners'])
out = dict(n=n, live_now=v(a), staged=v(b), staged_eq_pre_multiplier_code=v(b) == v(p) and b['score_sha256'] == p['score_sha256'],
           staged_all_equal=all(math.isclose(s, 1.0 / n, rel_tol=0, abs_tol=1e-15) for _, s in v(b)), staged_total=math.fsum(s for _, s in v(b)),
           same_uids=[u for u, _ in v(a)] == [u for u, _ in v(b)], formula=(a['formula']['version'], b['formula']['version']),
           floor=b['formula']['nonzero_weight_floor'], any_gain_key=any('gain' in k for m in b['miners'] for k in m),
           sha=(a['score_sha256'][:12], b['score_sha256'][:12]))
sys.path.insert(0, str(RA / '20261026-promotion-gate/code'))
tl = load('tpl', str(RA / '20261026-promotion-gate/code/training_params.py')); tn = load('tpn', str(R / 'stage/training_params.py'))
x, y = tl.build(), tn.build()
k1x, k1y = x['k1'] if 'k1' in x else x.get('training', {}).get('k1'), y['k1'] if 'k1' in y else y.get('training', {}).get('k1')
def strip(d):
    d = json.loads(json.dumps(d)); d.get('k1', {}).pop('king_gain_multiplier', None); return d
def diffkeys(p, q, path=''):
    if isinstance(p, dict) and isinstance(q, dict):
        r = []
        for k in set(p) | set(q): r += diffkeys(p.get(k), q.get(k), path + '/' + k)
        return r
    return [] if p == q else [path]
out['tp_live_has_kgm'] = 'king_gain_multiplier' in json.dumps(x); out['tp_staged_has_kgm'] = 'king_gain_multiplier' in json.dumps(y)
out['tp_other_diffs'] = diffkeys(strip(x), y)
print(json.dumps(out, indent=1))
