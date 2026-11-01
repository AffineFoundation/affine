"""Prior-weighted baseline selection (Jacob pt988u, 2026-11-01). NOT LIVE: nothing here is wired into gate_service.py.

Weight of a task = posterior expected Bernoulli variance under a Beta posterior that accounts for the sample count:
    a = s + a0_env,  b = (n - s) + b0_env,   w = E[p(1-p)] = a*b / ((a+b) * (a+b+1))
so a 1/1 task is NOT treated as p=1:  Beta(1,1) prior -> 1/1: 0.167, 16/16: 0.050, 64/64: 0.015, 0/0: 0.167, 8/16: 0.235.
Per-env prior (a0, b0): Beta(1,1) by default, or empirical Bayes ('eb'): method of moments on that env's tasks with n >= eb_min_n,
    m = mean(s/n), var_p = max(var(s/n) - mean(p(1-p)/n), tiny), kappa = m(1-m)/var_p - 1, kappa clipped to [kappa_min, kappa_max]
    (default [0.5, 4]: a weak prior, so 4 own trials already outweigh it), a0 = m*kappa, b0 = (1-m)*kappa
so a new task in an env starts at the env mean instead of 0.5.
Draw: fixed size n, stratified by env (each env gets at least min_share*n, the rest proportional to the env's weight mass), and
within an env probability-proportional-to-w sampling WITHOUT replacement (systematic PPS on a seeded random order, inclusion
probabilities pi_i = min(1, c*w_i) with sum pi = n_env exactly). No hard filters: every task has pi_i > 0. pi_i is recorded per task,
so Horvitz-Thompson / Hajek weighting (1/pi_i) recovers population (uniform-over-tasks) estimates.
Secret seed = os.urandom(32).hex(); commitment = sha256("<VERSION>|<seed hex>|<comma-joined sorted task ids>"), the same rule as
gate_core.sample_commitment with env-prefixed string ids."""
from __future__ import annotations
import hashlib, math, os, random

VERSION = "sn120-prior-select-v1"
DEFAULTS = dict(n=1000, min_share={"affine_math": 0.3, "dapo": 0.3}, prior="eb", kappa_min=0.5, kappa_max=4.0, eb_min_n=4,
                envs=None)


def post_var(s, n, a0=1.0, b0=1.0):
    a = s + a0; b = (n - s) + b0
    return a * b / ((a + b) * (a + b + 1.0))


def post_mean(s, n, a0=1.0, b0=1.0):
    return (s + a0) / (n + a0 + b0)


def env_priors(rows, mode="eb", kappa_min=0.5, kappa_max=4.0, min_n=4):
    """-> {env: dict(a0, b0, m, kappa, tasks_used, mode)}"""
    out = {}
    for env in sorted({r["env"] for r in rows}):
        rs = [r for r in rows if r["env"] == env and r["n"] >= min_n]
        if mode != "eb" or len(rs) < 30:
            out[env] = dict(a0=1.0, b0=1.0, m=0.5, kappa=2.0, tasks_used=len(rs), mode="uniform"); continue
        ps = [r["s"] / r["n"] for r in rs]; m = sum(ps) / len(ps)
        v = sum((p - m) ** 2 for p in ps) / max(1, len(ps) - 1)
        noise = sum(p * (1 - p) / r["n"] for p, r in zip(ps, rs)) / len(ps)
        m = min(max(m, 1e-3), 1 - 1e-3); vp = max(v - noise, 1e-6)
        kappa = min(max(m * (1 - m) / vp - 1.0, kappa_min), kappa_max)
        out[env] = dict(a0=m * kappa, b0=(1 - m) * kappa, m=m, kappa=kappa, kappa_raw=m * (1 - m) / vp - 1.0, tasks_used=len(rs), mode="eb")
    return out


def weights(rows, priors):
    return [post_var(r["s"], r["n"], priors[r["env"]]["a0"], priors[r["env"]]["b0"]) for r in rows]


def _alloc(n, mass, sizes, min_share):
    """Integer allocation per env: at least ceil(min_share*n) (capped by env size), the rest proportional to weight mass."""
    envs = sorted(mass); alloc = {e: min(sizes[e], math.ceil(float(min_share.get(e, 0.0)) * n)) for e in envs}
    rest = n - sum(alloc.values())
    while rest > 0:
        open_ = [e for e in envs if alloc[e] < sizes[e]]
        if not open_: break
        M = sum(mass[e] for e in open_) or 1.0
        add = {e: min(sizes[e] - alloc[e], int(rest * mass[e] / M)) for e in open_}
        if sum(add.values()) == 0:
            e = max(open_, key=lambda e: mass[e] / max(1, alloc[e])); add = {e: 1}
        for e, a in add.items(): alloc[e] += a
        rest = n - sum(alloc.values())
    return alloc


def _pps_probs(w, k):
    """pi_i = min(1, c*w_i), sum pi = k (iterative capping)."""
    k = min(k, len(w)); pi = [0.0] * len(w); fixed = set()
    while True:
        free = [i for i in range(len(w)) if i not in fixed]; kk = k - len(fixed)
        W = sum(w[i] for i in free)
        if kk <= 0 or W <= 0: break
        c = kk / W; over = [i for i in free if c * w[i] >= 1.0]
        if not over:
            for i in free: pi[i] = c * w[i]
            break
        fixed.update(over)
    for i in fixed: pi[i] = 1.0
    return pi


def _systematic(pi, rng):
    """Systematic PPS on a random order: selects exactly round(sum pi) units with first-order inclusion probability pi_i."""
    order = list(range(len(pi))); rng.shuffle(order)
    u = rng.random(); acc = 0.0; out = []; nxt = u
    for i in order:
        lo = acc; acc += pi[i]
        while nxt < acc - 1e-12:
            if nxt >= lo: out.append(i)
            nxt += 1.0
    return sorted(set(out))


def draw(rows, seed_hex, cfg=None):
    """rows: prior-table rows {task_id, env, s, n, ...}. -> dict(ids, cells{id: {env, pi, w, s, n, q}}, alloc, priors, ...)"""
    c = dict(DEFAULTS, **(cfg or {}))
    if not (isinstance(seed_hex, str) and len(seed_hex) == 64): raise ValueError("seed must be 64 hex chars")
    rows = sorted([r for r in rows if c["envs"] is None or r["env"] in c["envs"]], key=lambda r: r["task_id"])
    pri = env_priors(rows, c["prior"], c["kappa_min"], c["kappa_max"], c["eb_min_n"])
    w = weights(rows, pri)
    rng = random.Random(int(hashlib.sha256(("%s:%s:prior-sample" % (VERSION, seed_hex)).encode()).hexdigest(), 16))
    by_env = {}
    for r, x in zip(rows, w): by_env.setdefault(r["env"], []).append((r, x))
    mass = {e: sum(x for _, x in v) for e, v in by_env.items()}; sizes = {e: len(v) for e, v in by_env.items()}
    alloc = _alloc(int(c["n"]), mass, sizes, c["min_share"])
    cells = {}
    for e in sorted(by_env):
        rs = by_env[e]; pi = _pps_probs([x for _, x in rs], alloc[e])
        for i in _systematic(pi, rng):
            r, x = rs[i]; p = pri[e]
            cells[r["task_id"]] = dict(env=e, pi=pi[i], w=x, s=r["s"], n=r["n"], b=r.get("b", 0),
                                       q=post_mean(r["s"], r["n"], p["a0"], p["b0"]))
    ids = sorted(cells)
    return dict(version=VERSION, ids=ids, cells=cells, alloc=alloc, priors=pri, env_mass=mass, env_sizes=sizes,
                n=len(ids), cfg={k: v for k, v in c.items()})


def commitment(seed_hex, ids):
    return hashlib.sha256(("%s|%s|%s" % (VERSION, seed_hex, ",".join(sorted(ids)))).encode()).hexdigest()


def new_seed():
    return os.urandom(32).hex()


def hajek(values: dict, cells: dict):
    """Hajek (ratio HT) estimate of the population mean of values[t] over the stratified universe: sum(v/pi) / sum(1/pi)."""
    num = den = 0.0
    for t, v in values.items():
        if v is None: continue
        ip = 1.0 / cells[t]["pi"]; num += v * ip; den += ip
    return num / den if den else None
