#!/usr/bin/env python3
"""usage: export_params.py FIT.npz OUT.json [fit tag] - private IRT params for the live irt-info draw (mode 600, atomic replace).
Sanity checks before replacing: all pool tasks present, finite a > 0 and b, a median in [0.5, 6]; if OUT exists, Spearman(a_new, a_old) > 0.5."""
import json, os, sys, time, hashlib
import numpy as np
os.environ['TZ'] = 'America/Costa_Rica'; time.tzset()
src, out = sys.argv[1], sys.argv[2]; tag = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(src)
F = np.load(src, allow_pickle=False)
tasks = [str(t) for t in F['tasks']]; a = F['a'].astype(float); b = F['b'].astype(float); th = F['theta'].astype(float); cks = [str(c) for c in F['cks']]
pool = [json.loads(l)['id'] for l in open('/home/const/subnet120-rewrite/state/root-audits/20261101-baseline-switch/public/gate-pool-v1.jsonl')]
assert set(pool) <= set(tasks), 'pool tasks missing from fit'
assert np.isfinite(a).all() and (a > 0).all() and np.isfinite(b).all() and np.isfinite(th).all(), 'non-finite params'
med = float(np.median(a)); assert 0.5 <= med <= 6, 'a median %.3f out of range' % med
if os.path.exists(out):
    old = json.load(open(out)); oa = np.array([old['tasks'][t][0] for t in tasks])
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(oa)); rho = float(np.corrcoef(ra, rb)[0, 1])
    print('spearman a vs previous params: %.3f' % rho); assert rho > 0.5, 'refit a disagrees with previous params (rho %.3f)' % rho
# fallback theta: the most recent head known to the gate that is in the fit, else the top checkpoint
fb = float(th.max())
doc = dict(version='sn120-irt-params-v1', fit=tag, fitted_at=os.path.getmtime(src), fitted_at_cr=time.strftime('%Y-%m-%d %H:%M CR', time.localtime(os.path.getmtime(src))),
           exported_at_cr=time.strftime('%Y-%m-%d %H:%M CR'), ckpts=len(cks), a_median=med, theta_fallback=fb,
           ckpt_theta={c: round(float(t), 5) for c, t in zip(cks, th)},
           tasks={t: [round(float(x), 5), round(float(y), 5)] for t, x, y in zip(tasks, a, b)})
os.makedirs(os.path.dirname(out), mode=0o700, exist_ok=True)
tmp = out + '.tmp'; fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, 'w') as f: json.dump(doc, f, sort_keys=True)
os.replace(tmp, out); os.chmod(out, 0o600)
print('wrote %s tasks %d ckpts %d a_median %.3f theta_fallback %.3f sha256 %s' % (out, len(tasks), len(cks), med, fb, hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16]))
