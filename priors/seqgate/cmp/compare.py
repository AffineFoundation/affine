"""SN120 gate test-design comparison at <= today's task budget (Jacob 2026-11-02 11:48 CR redirect). OFFLINE only, read-only on live
state; private outputs in RA/20261102-seqgate-shadow/private/cmp/. Designs:
  A  today: E[N]=500 prior-v1 draw, head 12 / cand 8, per-env Hajek, z>1 (+ sigma_run_floor 0.003), gain>0
  B  CUPED: residual d_i - theta_e (X_i - mu_X,e), X = [IRT P_i(theta_head), prior-table posterior mean], mu_X exact over the pool,
     theta cross-fitted (2 folds, weighted LS by 1/pi)
  C  allocation by head stratum + discordance: candidate k_i = 8 on head-mixed tasks, k_lo on head-extreme tasks (head 12/12 or 0/12);
     extra k_x rollouts on head-extreme tasks whose first candidate rollouts disagree with the head (prob 1) or agree (prob pi0),
     unbiased IPW correction est = p1 + Z/pi (p2 - p1)
  D  group-sequential looks (default 300/500/700 of an E[N]=700 draw), GREEN boundary b_k = c / sqrt(t_k) (OBF) or c (Pocock),
     futility RED if z_k < f at an interim look; c calibrated on the simulated null to the target false-GREEN rate
  P  paired seeds (CRN): head rollout j and candidate rollout j share the request seed; modeled as a shared uniform with prob rho
Truth model (synthetic): pool = 19,305 gate tasks; head p_i = sigmoid(a_i (theta_h - b_i)) from the live 2PL fit; candidate logit
= a_i (theta_h + s - b_i) + eps_i, eps_i ~ N(0, s_eps^2) per replicate (s_eps calibrated to the real per-task gain spread); s solved
so the pool-weighted Delta hits the target exactly. Cost unit = candidate task-equivalent = candidate rollouts / 8."""
import json, math, os, sys, sqlite3, importlib.util, time
import numpy as np
from pathlib import Path
RA = Path('/home/const/subnet120-rewrite/state/root-audits'); G = RA / '20261026-promotion-gate'; ST = G / 'state'
OUT = RA / '20261102-seqgate-shadow/private/cmp'; OUT.mkdir(parents=True, exist_ok=True); os.chmod(OUT.parent, 0o700)
spec = importlib.util.spec_from_file_location('bv_ro', G / 'code/baseline_v1.py'); BV = importlib.util.module_from_spec(spec); spec.loader.exec_module(BV)
rj = lambda p: json.load(open(p))
IRTP = RA / '20261102-irt-baseline/private/live/irt_params.private.json'
sig = lambda x: 1 / (1 + np.exp(-x))


def population(rid=None):
    rid = rid or sorted(x for x in os.listdir(ST / 'heldout') if x.startswith('r2026'))[-1]
    d = ST / 'heldout' / rid; r = rj(d / 'ROUND.private.json'); PI = rj(d / 'PI.private.json')
    ids = sorted(PI['pi']); env = np.array([0 if PI['env'][t] == 'affine_math' else 1 for t in ids])
    rows = BV.load_rows(str(d / 'PRIOR-SNAPSHOT.private.sqlite'), ids)
    s = np.array([x['s'] for x in rows]); n = np.array([x['n'] for x in rows])
    P = rj(IRTP); a = np.array([P['tasks'][t][0] for t in ids]); b = np.array([P['tasks'][t][1] for t in ids])
    B = rj(d / 'BASELINE.private.json')['per_task']
    cells = [(t, int(v['n']), int(v['c'])) for t, v in B.items()]
    th, se, nr = BV.theta_map(cells, {t: (P['tasks'][t][0], P['tasks'][t][1]) for t in ids})
    rws = [dict(task_id=t, env=PI['env'][t]) for t in ids]
    def irt_pi(clip=None):
        out = []
        for t in ids:
            aa, bb = P['tasks'][t]; p = BV._sig(aa * (th - bb))
            if clip: p = min(max(p, clip[0]), clip[1])
            out.append(BV.irt_a_used(aa, P['a_median']) * math.sqrt(p * (1 - p)))
        pi = BV.inclusion_from(rws, out, N=500, floor=0.10, cap=0.10); return np.array([pi[t] for t in ids])
    pv = r['cfg']; pool = r['pool']['per_env']
    return dict(rid=rid, ids=ids, env=env, a=a, b=b, theta=th, prior=(s + 1) / (n + 2), prior_n=n,
                pi=dict(prior=np.array([PI['pi'][t] for t in ids]), irt=irt_pi(), irtclip=irt_pi((0.05, 0.95))),
                sizes=np.array([pool['affine_math'], pool['dapo']]), head_obs={t: (v['n'], v['c']) for t, v in B.items()})


def solve_shift(Pp, eps, target):
    lo, hi = -3.0, 3.0; base = sig(Pp['a'] * (Pp['theta'] - Pp['b']))
    for _ in range(50):
        m = (lo + hi) / 2; dl = np.mean(sig(Pp['a'] * (Pp['theta'] + m - Pp['b']) + eps) - base)
        lo, hi = (m, hi) if dl < target else (lo, m)
    return (lo + hi) / 2


# ------------------------------------------------------------------ estimator
def hajek(d, pi, env, sizes, v=None, X=None, muX=None, floor=0.0):
    """per-env Hajek (size shares) of d; optional CUPED residualization with cross-fitted theta. -> (est, se)"""
    sh = sizes / sizes.sum(); est = 0.0; var = 0.0
    for e in (0, 1):
        m = env == e
        if m.sum() < 4: continue
        de, pe = d[m].copy(), pi[m]; we = 1 / pe
        if X is not None:
            Xe = X[m] - muX[e]; fold = (np.arange(m.sum()) % 2).astype(bool); adj = np.zeros(m.sum())
            for f in (False, True):
                tr = fold != f; A = Xe[tr] * we[tr, None]
                th = np.linalg.lstsq(np.c_[np.ones(tr.sum()), Xe[tr]] * np.sqrt(we[tr, None]), de[tr] * np.sqrt(we[tr]), rcond=None)[0][1:]
                adj[fold == f] = Xe[fold == f] @ th
            de = de - adj
        Ne = we.sum(); me = (we * de).sum() / Ne
        ve = v[m] if v is not None else 0.0
        Ve = (((1 - pe) * (de - me) ** 2 + pe * ve) / pe ** 2).sum() / Ne ** 2
        est += sh[e] * me; var += sh[e] ** 2 * Ve
    return est, math.sqrt(var + floor ** 2)


# ------------------------------------------------------------------ one replicate -> per-task arrays in visit order
def replicate(Pp, rng, target, s_eps, wkey='prior', Nmax=1000, rho=0.0, kc=8, klo=2, kx=6, pi0=0.15):
    eps = rng.normal(0, s_eps, len(Pp['ids'])) if s_eps > 0 else np.zeros(len(Pp['ids']))
    sft = solve_shift(Pp, eps, target)
    ph = sig(Pp['a'] * (Pp['theta'] - Pp['b'])); pc = sig(Pp['a'] * (Pp['theta'] + sft - Pp['b']) + eps)
    true = float(np.average(pc - ph, weights=np.ones_like(ph)))   # size shares -> uniform pool weights
    pi = Pp['pi'][wkey] * (Nmax / 500.0)
    sel = np.nonzero(rng.random(len(pi)) < pi)[0]; rng.shuffle(sel); n = len(sel)
    ph_, pc_ = ph[sel], pc[sel]
    Uh = rng.random((n, 12)); Uc = rng.random((n, kc + kx))
    if rho > 0:   # paired seeds: rollout j of head and candidate share the uniform with prob rho
        share = rng.random((n, 8)) < rho; Uc[:, :8] = np.where(share, Uh[:, :8], Uc[:, :8])
    hc = (Uh < ph_[:, None]).sum(1); h = hc / 12
    cr = (Uc < pc_[:, None])
    c8 = cr[:, :8].sum(1) / 8
    # C: stratified + discordance
    ext = (hc == 12) | (hc == 0)
    p1 = np.where(ext, cr[:, :klo].sum(1) / klo, c8)
    disc = ext & (np.abs(p1 - h) > 0)
    pz = np.where(disc, 1.0, pi0); Z = ext & (rng.random(n) < pz)
    p2 = cr[:, kc:kc + kx].sum(1) / kx
    cC = np.where(ext, p1 + np.where(Z, (p2 - p1) / pz, 0.0), c8)
    costC = np.where(ext, klo + Z * kx, 8) / 8.0
    vh = h * (1 - h) / 11
    vC = np.where(ext, np.where(Z, p2 * (1 - p2) / max(1, kx - 1) / pz ** 2 * 1.0, p1 * (1 - p1) / max(1, klo - 1)), c8 * (1 - c8) / 7)
    X = np.c_[ph_ * 0 + sig(Pp['a'][sel] * (Pp['theta'] - Pp['b'][sel])), Pp['prior'][sel]]
    return dict(true=true, n=n, env=Pp['env'][sel], pi=pi[sel], d=c8 - h, v=c8 * (1 - c8) / 7 + vh, dC=cC - h, vC=vC + vh, costC=costC, X=X)


def muX(Pp):
    ph = sig(Pp['a'] * (Pp['theta'] - Pp['b'])); out = []
    for e in (0, 1):
        m = Pp['env'] == e; out.append(np.array([ph[m].mean(), Pp['prior'][m].mean()]))
    return out


DESIGNS = {
    # name: (per-task protocol, CUPED?, looks as task counts of the E[N]=700 visit order (None = fixed prefix), fixed-n prefix)
    'A_today': ('std', False, None, 500), 'A_700': ('std', False, None, 700), 'B_cuped': ('std', True, None, 500), 'C_strat': ('C', False, None, None),
    'BC': ('C', True, None, None), 'D_gs': ('std', False, (300, 500, 700), None), 'BD': ('std', True, (300, 500, 700), None),
    'BCD': ('C', True, (300, 500, 700), None), 'CD': ('C', False, (300, 500, 700), None), 'D_gs2': ('std', False, (300, 600), None),
    'CD2': ('C', False, (300, 600), None), 'A_nofloor': ('std', False, None, 500),
    'D_369': ('std', False, (300, 600, 900), None), 'D_4710': ('std', False, (400, 700, 1000), None), 'D_25': ('std', False, (250, 500), None),
    'A_1000': ('std', False, None, 1000),
}
if os.environ.get('ONLY'): DESIGNS = {k: v for k, v in DESIGNS.items() if k in os.environ['ONLY'].split(',')}


def zpath(R, Pp, MX, design, floor, Nmax=1000, budgetC=None):
    proto, cup, looks, fixn = DESIGNS[design]
    d, v, cost = (R['dC'], R['vC'], R['costC']) if proto == 'C' else (R['d'], R['v'], np.ones(R['n']))
    n = R['n']
    if looks is None:
        if proto == 'C':   # prefix sized so the mean cost matches today's 8 x E[N]=500
            cum = np.cumsum(cost); m = int(np.searchsorted(cum, budgetC * n / Nmax)); m = max(m, 50)
        else: m = int(round(n * fixn / Nmax))
        ms = [m]
    elif proto == 'C':   # looks in candidate task-equivalents (cost), not tasks
        cum = np.cumsum(cost); ms = [max(50, int(np.searchsorted(cum, L * n / Nmax))) for L in looks]
    else: ms = [int(round(n * L / Nmax)) for L in looks]
    zs, gs, cs = [], [], []
    for m in ms:
        sl = slice(0, m)
        est, se = hajek(d[sl], R['pi'][sl], R['env'][sl], Pp['sizes'], v[sl], R['X'][sl] if cup else None, MX if cup else None, floor)
        zs.append(est / se if se > 0 else 0.0); gs.append(est); cs.append(float(cost[sl].sum()))
    return zs, gs, cs


def decide(zs, gs, cs, rule):
    """rule: dict(kind='fixed', c) | dict(kind='gs', c, shape, fut) -> (green, cost)"""
    K = len(zs)
    if rule['kind'] == 'fixed': return (zs[0] > rule['c'] and gs[0] > 0), cs[0]
    t = np.array(rule['tfrac'])
    for k in range(K):
        b = rule['c'] / math.sqrt(t[k]) if rule['shape'] == 'obf' else rule['c']
        if zs[k] > b and gs[k] > 0: return True, cs[k]
        if k < K - 1 and zs[k] < rule['fut'][k]: return False, cs[k]
    return False, cs[-1]


def calibrate(paths, rule, alpha):
    lo, hi = -1.0, 6.0
    for _ in range(40):
        m = (lo + hi) / 2; r = dict(rule, c=m)
        fg = np.mean([decide(*p, r)[0] for p in paths])
        lo, hi = (lo, m) if fg <= alpha else (m, hi)
    return hi


def main():
    reps0 = int(os.environ.get('REPS0', 3000)); reps1 = int(os.environ.get('REPS1', 1000)); t0 = time.time()
    Pp = population(); MX = muX(Pp); rng = np.random.default_rng(20261102)
    s_eps = float(os.environ.get('S_EPS', 0.0)); floor = 0.003
    targets = [0.0, 0.003, 0.005, 0.01, -0.005, -0.01]
    budgetC = 500.0
    res = dict(rid=Pp['rid'], theta=Pp['theta'], s_eps=s_eps, reps0=reps0, reps1=reps1, runs={})
    runs = (('prior', 0.0), ('prior', float(os.environ.get('RHO', 0.08))), ('irt', 0.0), ('irtclip', 0.0))
    if os.environ.get('RUNS'): runs = [r for r in runs if '%s-%.2f' % r in os.environ['RUNS'].split(',')]
    outname = os.environ.get('OUTNAME', 'COMPARE')
    for wkey, rho in runs:
        tag = '%s-rho%.2f' % (wkey, rho)
        paths = {}
        for tg in targets:
            reps = reps0 if tg == 0.0 else reps1; P = {k: [] for k in DESIGNS}; trues = []
            for i in range(reps):
                R = replicate(Pp, rng, tg, s_eps, wkey, rho=rho); trues.append(R['true'])
                for k in DESIGNS: P[k].append(zpath(R, Pp, MX, k, floor if k == 'A_today' else 0.0, budgetC=budgetC))
                if wkey != 'prior' or rho > 0:
                    if i >= (reps // 3): break   # other weightings / CRN: fewer reps
            paths[tg] = P; print(tag, tg, 'reps', len(P['A_today']), 'true', round(float(np.mean(trues)), 5), round(time.time() - t0), flush=True)
        # operating points
        today = dict(kind='fixed', c=1.0)
        a0 = float(np.mean([decide(*p, today)[0] for p in paths[0.0]['A_today']]))
        ops = {}
        for alpha_name, alpha in (('today', a0), ('a05', 0.05)):
            for k, (proto, cup, looks, fixn) in DESIGNS.items():
                if looks is None: rules = [('fixed', dict(kind='fixed'))]
                else:
                    tf = [L / looks[-1] for L in looks]
                    rules = [('%s-fut%s' % (sh, fu), dict(kind='gs', shape=sh, tfrac=tf, fut=[fu] * (len(looks) - 1))) for sh in ('obf', 'pocock') for fu in (-9.0, 0.0, 0.5)]
                for rn, rule in rules:
                    if k == 'A_today' and alpha_name == 'today': rule = today
                    else: rule = dict(rule, c=calibrate(paths[0.0][k], rule, alpha))
                    row = dict(c=rule['c'])
                    for tg in targets:
                        dd = [decide(*p, rule) for p in paths[tg][k]]
                        row['green_%g' % tg] = float(np.mean([x[0] for x in dd])); row['tasks_%g' % tg] = float(np.mean([x[1] for x in dd]))
                    ops['%s|%s|%s' % (alpha_name, k, rn)] = row
        # unadjusted SE of today's estimator (duongna's SE claim): sd of the estimate under the null
        sd = float(np.std([p[1][0] for p in paths[0.0]['A_today']])); sdB = float(np.std([p[1][0] for p in paths[0.0]['B_cuped']])) if 'B_cuped' in DESIGNS else None
        res['runs'][tag] = dict(alpha_today=a0, ops=ops, sd_gain_A=sd, sd_gain_B=sdB)
        json.dump(res, open(OUT / (outname + '.private.json'), 'w')); os.chmod(OUT / (outname + '.private.json'), 0o600)
    print('done', round(time.time() - t0))


if __name__ == '__main__': main()
