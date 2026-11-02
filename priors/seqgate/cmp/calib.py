"""calibration inputs for compare.py from real data (private outputs): per-task gain spread tau^2 on the 59 prior-v1 turns,
paired-seed outcome correlation from the A/A run (arm C reuses arm A's seeds), real-turn CUPED variance reduction (prior-table X,
round's own snapshot = pre-treatment; IRT X reported separately: the live fit may include these checkpoints' cells)."""
import json, os, math, collections, numpy as np, importlib.util
from pathlib import Path
import compare as CP
RA = CP.RA; ST = CP.ST; BV = CP.BV; rj = CP.rj
out = {}
# ---- tau^2 and CUPED on real turns
P = rj(CP.IRTP); tau = []; red = collections.defaultdict(list); ext = []; irtcal = []
for e in sorted(int(x) for x in os.listdir(ST / 'epochs') if x.isdigit()):
    d = ST / 'epochs' / str(e); D = rj(d / 'DECISION.json') if (d / 'DECISION.json').exists() else None
    if not D or (D.get('heldout') or {}).get('baseline') != 'prior-v1' or not (d / 'C-PROBE.private.json').exists(): continue
    rid = D['heldout']['round_id']; rd = ST / 'heldout' / rid; r = rj(rd / 'ROUND.private.json'); PI = rj(rd / 'PI.private.json'); pv = r['cfg']
    H = {BV.task_of(int(k), pv): v for k, v in rj(d / 'H-PROBE.private.json')['per_task'].items()}
    C = {BV.task_of(int(k), pv): v for k, v in rj(d / 'C-PROBE.private.json')['per_task'].items()}
    ids = [t for t in r['ids'] if t in H and t in C and H[t]['n'] and C[t]['n']]
    rows = {x['task_id']: x for x in BV.load_rows(str(rd / 'PRIOR-SNAPSHOT.private.sqlite'), ids)}
    th = P['ckpt_theta'].get(r['head'][:8], P['theta_fallback'])
    for envn in ('affine_math', 'dapo'):
        T = [t for t in ids if PI['env'][t] == envn]
        if len(T) < 20: continue
        w = np.array([1 / PI['pi'][t] for t in T]); x = np.array([C[t]['c'] / C[t]['n'] for t in T]); y = np.array([H[t]['c'] / H[t]['n'] for t in T])
        dd = x - y; nv = x * (1 - x) / (np.array([C[t]['n'] for t in T]) - 1) + y * (1 - y) / (np.array([H[t]['n'] for t in T]) - 1)
        m = np.average(dd, weights=w); tau.append((len(T), float(np.average((dd - m) ** 2, weights=w) - np.average(nv, weights=w)), envn))
        ext.append(float(np.average(((y == 1) | (y == 0)).astype(float), weights=w)))
        Xp = np.array([(rows[t]['s'] + 1) / (rows[t]['n'] + 2) for t in T])
        Xi = np.array([1 / (1 + math.exp(-P['tasks'][t][0] * (th - P['tasks'][t][1]))) for t in T])
        irtcal.append((float(np.average(Xi, weights=w)), float(np.average(y, weights=w))))
        for name, X in (('prior', Xp[:, None]), ('irt', Xi[:, None]), ('prior+irt', np.c_[Xp, Xi]), ('head_obs', y[:, None])):
            # cross-fitted residual variance ratio (mu_X from the sample itself: ratio of weighted residual variance only)
            fold = np.arange(len(T)) % 2 == 1; res = dd.copy()
            for f in (False, True):
                tr = fold != f; A = np.c_[np.ones(tr.sum()), X[tr]] * np.sqrt(w[tr, None])
                b = np.linalg.lstsq(A, dd[tr] * np.sqrt(w[tr]), rcond=None)[0]
                res[fold == f] = dd[fold == f] - np.c_[np.ones((fold == f).sum()), X[fold == f]] @ b + b[0]
            red[name].append(float(np.average((res - np.average(res, weights=w)) ** 2, weights=w) / np.average((dd - m) ** 2, weights=w)))
out['tau2'] = dict(mean=float(np.average([t[1] for t in tau], weights=[t[0] for t in tau])), by_env={en: float(np.mean([t[1] for t in tau if t[2] == en])) for en in ('affine_math', 'dapo')}, n=len(tau))
out['cuped_real_var_ratio'] = {k: dict(mean=float(np.mean(v)), median=float(np.median(v)), n=len(v)) for k, v in red.items()}
out['head_extreme_share'] = float(np.mean(ext)); out['irt_calibration_pred_vs_obs'] = [float(np.mean([a for a, b in irtcal])), float(np.mean([b for a, b in irtcal]))]
# ---- paired seeds: A/A run arm A vs arm C (same request seeds) vs arm B (independent), per (round, task) k=2 each
A = RA / '20261101-aa-noise/private/run2/rollouts.jsonl'; X = collections.defaultdict(dict)
for l in open(A):
    q = json.loads(l)
    if q['arm'] in 'ABC': X[(q['round'], q['task_id'], q['j'])][q['arm']] = q['kind'] == 'positive'
both = [v for v in X.values() if len(v) == 3]
ac = np.array([[v['A'], v['C']] for v in both], float); ab = np.array([[v['A'], v['B']] for v in both], float)
def corr(m): return float(np.corrcoef(m[:, 0], m[:, 1])[0, 1])
# within-task correlation: subtract task means (p varies by task -> raw corr is inflated by task difficulty)
tk = collections.defaultdict(list)
for k, v in X.items():
    if len(v) == 3: tk[k[1]].append(v)
def within(a1, a2):
    num = den1 = den2 = 0.0
    for t, L in tk.items():
        x = np.array([v[a1] for v in L], float); y = np.array([v[a2] for v in L], float); pm = (x.mean() + y.mean()) / 2
        num += ((x - pm) * (y - pm)).sum(); den1 += ((x - pm) ** 2).sum(); den2 += ((y - pm) ** 2).sum()
    return num / math.sqrt(den1 * den2)
out['paired_seed'] = dict(pairs=len(both), within_task_corr_shared=within('A', 'C'), within_task_corr_indep=within('A', 'B'),
                          note='same checkpoint both arms (head vs itself); head vs a different candidate will share less')
json.dump(out, open(CP.OUT / 'CALIB.private.json', 'w'), indent=1); os.chmod(CP.OUT / 'CALIB.private.json', 0o600)
print(json.dumps(out, indent=1))
