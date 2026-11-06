"""SN120 prior-table baseline v1 (Jacob Steeves, 2026-11-01): the math of the gate's per-round baseline draw and estimate.
Pure functions (no pod / gate I/O). Public code; the prior TABLE values it reads are private.

Prior (per task i of the public gate pool, from the prior table that a dedicated pod keeps filling with the current head):
  s_i / n_i = (effective) solved / sampled counts of the head. On a head change every row is discounted x0.5 (old evidence still
  counts, with half the weight). A task with no samples yet (n = 0) starts from its env average: s = m_env * n_init, n = n_init.
  Posterior p_i ~ Beta(s_i + 1, n_i - s_i + 1);  v_i = E[p_i (1 - p_i)] = A B / ((A + B)(A + B + 1)), A = s + 1, B = n - s + 1.
Draw (per round, Poisson sampling, expected size N):
  w_i = share_env / |env| (share = env size / pool size -> w_i = 1/|pool|; or equal env shares)
  a_i = w_i * sqrt(v_i)                        (Neyman allocation)
  pi_i = N * ((1 - floor) * a_i / sum a + floor * w_i / sum w)    (floor = uniform part, default 0.10)
  pi_i capped at `cap` (default 0.10), the excess re-spread over the uncapped tasks until no pi_i exceeds the cap.
  weighting = 'irt-info' (live from 2026-11-02): sqrt(v_i) is replaced by sqrt(I_i), the 2PL Fisher information of task i at the
  head's ability (see irt_score below); w, N, floor and cap unchanged.
  task i is drawn iff u_i < pi_i, u_i = int(sha256(VERSION|draw|seed|task_id)[:16], 16) / 2^64.
Seed (commit-reveal): secret = 32 random bytes; secret_commitment = sha256(VERSION|secret|<secret hex>) is published when the round
  opens; after the upload deadline the gate fixes a future block number B (published before B exists);
  seed = sha256(VERSION|seed|<secret hex>|B|<hash of block B>). The secret is revealed after the round.
Estimate (per turn): every drawn task is scored fresh: head h_i = c/n of the head's rollouts, candidate x_i = c/n of its k rollouts,
  d_i = x_i - h_i (the turn batch is not excluded). Per env e (Hajek / inverse-probability ratio estimator):
    m_e = sum_{i in e} d_i / pi_i / sum_{i in e} 1 / pi_i
    Delta = sum_e share_e * m_e      (estimates the pool-wide mean of p_cand - p_head)
    Var_e = sum_{i in e} ((1 - pi_i) r_i^2 + pi_i v_i) / pi_i^2 / (sum_{i in e} 1/pi_i)^2,  r_i = d_i - m_e,
            v_i = x(1-x)/(k-1) + h(1-h)/(m-1)   (rollout noise)
    run term (head and candidate are separate fresh runs; a run can shift all its pass rates a little):
            Var_run = max(tau^2 * 2 * A^2, sigma_run_floor^2), A = sum_e share_e * Hajek_e(h(1-h) m/(m-1));
            live: tau = 0 and sigma_run_floor = 0 (no run-noise floor) in rounds opened after the 2026-11-03 change (Jacob pt1180u;
            0.003 before; the 2026-11-01 head-vs-itself test found no measurable run noise), so Var_run = 0 and
            SE = sqrt(sum_e share_e^2 Var_e). The round's value is frozen at open: ROUND cfg / settings.sigma_run_floor.
    SE = sqrt(sum_e share_e^2 Var_e + Var_run);  z = Delta / SE
  Request seeds: head block b uses plan_seed(seed, b); the candidate of epoch e uses cand_seed(seed, e) (independent seeds).
  GREEN iff Delta > 0 AND z > 1.00 (bar_z) AND candidate cap-hit < cap_max (CAP_HIT_MAX = 0.33) AND >= min_scored tasks scored.
  cap-hit (task-level, Jacob 2026-11-01 19:43 CR): the share of ALL the candidate's rollouts on the scored baseline tasks that hit
  the token cap (the round's max_tokens = caps.json max_output_tokens, 65,536 tokens, https://affine.io/caps.json; no EOS) = sum_i capped_i / sum_i rollouts_i, every scored task counted alike (affine_math and dapo), no env distinction.
  RED if >= 33% of the candidate's baseline answers hit the token cap. (Before E507: Hajek capped share on affine_math < 0.20.)
  The head's rate is computed the same way and recorded (h_cap_rate); per-env rates are diagnostics only (per_env).
  z bar: fixed at z > 1.00 (bar_z = 1.0) from E553 (Jacob 2026-11-02 08:20 CR); E513-E552 used a time-relaxed bar (the exact
  bar of every decision is in its thresholds.z_min)."""
from __future__ import annotations
import hashlib, json, math, os, sqlite3

VERSION = 'sn120-prior-baseline-v1'
CAP_HIT_MAX = 0.33   # task-level cap rule: RED if >= 33% of the candidate's baseline answers hit the token cap
ROUND_SCHEMA = 'sn120-heldout-gate-prior-v1'
DEFAULTS = dict(N=500, k_cand=8, head_rollouts=12, head_blocks=3, scoring_mode='separate', floor=0.10, cap=0.10, shares='size',
                n_init=2.0, tau_run_separate=0.0, tau_run_same=0.0, sigma_run_floor=0.0, bar_z=1.0, z_inclusive=False, cap_max=CAP_HIT_MAX,
                cap_rule='task-level', min_scored=100, min_cover=0.95, block_delay=3, baseline_parallel=3,
                stall_s=2100, probe_env='affine_gatepool', math_offset=0, dapo_offset=7496)


# ------------------------------------------------------------------ prior table (read-only)
def snapshot_table(src: str, dst: str) -> dict:
    """Consistent read-only snapshot of the live prior table (sqlite online backup; the source is opened mode=ro) -> dst (0600)."""
    s = sqlite3.connect('file:%s?mode=ro' % src, uri=True, timeout=60)
    fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600); os.close(fd)
    d = sqlite3.connect(dst)
    with d: s.backup(d)
    s.close(); d.close(); os.chmod(dst, 0o600)
    h = hashlib.sha256()
    with open(dst, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return dict(path=dst, sha256=h.hexdigest(), bytes=os.path.getsize(dst))


def load_rows(snapshot: str, pool_ids) -> list:
    """Prior rows {task_id, env, s, n, b} for the tasks of the public gate pool (tasks missing from the table get s = n = 0)."""
    con = sqlite3.connect('file:%s?mode=ro' % snapshot, uri=True)
    got = {r[0]: r for r in con.execute('SELECT task_id, env, s, n, b FROM tasks')}
    con.close(); out = []
    for t in pool_ids:
        env = 'affine_math' if t.startswith('math:') else 'dapo' if t.startswith('dapo:') else None
        r = got.get(t)
        out.append(dict(task_id=t, env=(r[1] if r else env), s=float(r[2]) if r else 0.0, n=float(r[3]) if r else 0.0,
                        b=float(r[4] or 0.0) if r else 0.0))
    return out


def table_digest(rows) -> str:
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r['task_id']): h.update(('%s|%.6f|%.6f\n' % (r['task_id'], r['s'], r['n'])).encode())
    return h.hexdigest()


# ------------------------------------------------------------------ inclusion probabilities
def env_shares(rows, shares='size') -> tuple:
    cnt = {}
    for r in rows: cnt[r['env']] = cnt.get(r['env'], 0) + 1
    P = sum(cnt.values())
    sh = {e: (c / P if shares == 'size' else 1.0 / len(cnt)) for e, c in cnt.items()}
    return sh, cnt


def post_var(s, n):
    A = s + 1.0; B = n - s + 1.0
    return A * B / ((A + B) * (A + B + 1.0))


def inclusion(rows, N=500, floor=0.10, cap=0.10, shares='size', n_init=2.0) -> dict:
    """-> {task_id: pi}. rows = prior rows of the gate pool. prior-v1 weighting: score_i = sqrt(E[p_i(1-p_i)])."""
    sh, cnt = env_shares(rows, shares)
    tot = {}
    for r in rows:
        if r['n'] > 0: t = tot.setdefault(r['env'], [0.0, 0.0]); t[0] += r['s']; t[1] += r['n']
    m = {e: (v[0] / v[1] if v[1] > 0 else 0.5) for e, v in tot.items()}
    score = []
    for r in rows:
        s, n = (r['s'], r['n']) if r['n'] > 0 else (m.get(r['env'], 0.5) * n_init, n_init)
        score.append(math.sqrt(post_var(s, n)))
    return inclusion_from(rows, score, N, floor, cap, shares)


def inclusion_from(rows, score, N=500, floor=0.10, cap=0.10, shares='size') -> dict:
    """-> {task_id: pi} for any per-task draw score (same order as rows): a_i = w_i * score_i,
    pi_i = N * ((1 - floor) * a_i / sum a + floor * w_i / sum w), capped at cap with the excess re-spread."""
    sh, cnt = env_shares(rows, shares)
    w = [sh[r['env']] / cnt[r['env']] for r in rows]; a = [wi * si for wi, si in zip(w, score)]
    P = len(rows); cv = min(float(cap) if cap else 1.0, 1.0)
    if cv * P < N * 1.0001: raise ValueError('cap %.3f infeasible for N=%d over %d tasks' % (cv, N, P))
    Sa, Sw = sum(a), sum(w)
    pi = [N * ((1 - floor) * ai / Sa + floor * wi / Sw) for ai, wi in zip(a, w)]
    for _ in range(500):
        over = [i for i, p in enumerate(pi) if p > cv]
        if not over: break
        ex = sum(pi[i] - cv for i in over)
        for i in over: pi[i] = cv
        free = [i for i, p in enumerate(pi) if p < cv]; F = sum(pi[i] for i in free)
        for i in free: pi[i] *= 1 + ex / F
    return {r['task_id']: p for r, p in zip(rows, pi)}


# ------------------------------------------------------------------ irt-info weighting (live from 2026-11-02, Jacob Steeves)
# 2PL item response model fitted on every pass/fail the network has recorded (checkpoints x gate-pool tasks):
#   P_i(theta) = 1 / (1 + exp(-a_i (theta - b_i)))   (a_i discrimination, b_i difficulty, theta = checkpoint ability, sd 1 over checkpoints)
# Draw score at the current head (theta_h): score_i = sqrt(I_i), I_i = a'_i^2 P_i(theta_h) (1 - P_i(theta_h)) (Fisher information),
#   a'_i = a_i shrunk toward the pool median m (log a' = (1 - a_shrink) log a + a_shrink log m) and clipped to [a_lo m, a_hi m],
#   because a_i is weakly determined for many tasks. Same N, floor and cap as prior-v1 (the 10% uniform floor keeps every task drawable).
# theta_h: 1-D MAP of the head's own pass/fail cells (its candidate probe + earlier head runs on the gate pool) under the fitted a, b
#   with a N(0, theta_prior_sd^2) prior; refreshed at every round open. a, b: refit daily; per-task a, b values are private.
IRT_DEFAULTS = dict(a_shrink=0.5, a_lo=0.5, a_hi=2.0, theta_prior_sd=2.0, theta_min_rollouts=200)


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def irt_a_used(a_raw, a_med, a_shrink=0.5, a_lo=0.5, a_hi=2.0):
    la = (1.0 - a_shrink) * math.log(max(a_raw, 1e-6)) + a_shrink * math.log(a_med)
    return min(max(math.exp(la), a_lo * a_med), a_hi * a_med)


def theta_map(cells, ab, prior_sd=2.0, iters=60):
    """cells = [(task_id, n, solved)], ab = {task_id: (a, b)} -> (theta, se, rollouts). Newton steps on the 1-D log posterior."""
    cl = [(ab[t][0], ab[t][1], float(n), float(c)) for t, n, c in cells if t in ab and n]
    th = 0.0; tau = 1.0 / (prior_sd * prior_sd); h = tau
    for _ in range(iters):
        g = -th * tau; h = tau
        for a, b, n, c in cl:
            p = _sig(a * (th - b)); g += a * (c - n * p); h += a * a * n * p * (1 - p)
        step = max(-1.0, min(1.0, g / h)); th += step
        if abs(step) < 1e-7: break
    return th, 1.0 / math.sqrt(h), int(sum(n for _, _, n, _ in cl))


def irt_score(rows, ab, theta, a_med, a_shrink=0.5, a_lo=0.5, a_hi=2.0) -> list:
    """sqrt(Fisher information) per task at theta (same order as rows); P from the fitted a, b, the multiplier from the shrunk a'."""
    out = []
    for r in rows:
        a, b = ab[r['task_id']]; p = _sig(a * (theta - b)); au = irt_a_used(a, a_med, a_shrink, a_lo, a_hi)
        out.append(au * math.sqrt(p * (1.0 - p)))
    return out


# ------------------------------------------------------------------ commit-reveal seed and draw
def new_secret() -> str:
    return os.urandom(32).hex()


def secret_commitment(secret: str) -> str:
    return hashlib.sha256(('%s|secret|%s' % (VERSION, secret)).encode()).hexdigest()


def seed_from(secret: str, block_number: int, block_hash: str) -> str:
    return hashlib.sha256(('%s|seed|%s|%d|%s' % (VERSION, secret, int(block_number), block_hash.lower())).encode()).hexdigest()


def uniform(seed: str, task_id: str) -> float:
    return int(hashlib.sha256(('%s|draw|%s|%s' % (VERSION, seed, task_id)).encode()).hexdigest()[:16], 16) / 2.0 ** 64


def draw(pi: dict, seed: str) -> list:
    return sorted(t for t, p in pi.items() if uniform(seed, t) < p)


def draw_commitment(seed: str, ids) -> str:
    return hashlib.sha256(('%s|%s|%s' % (VERSION, seed, ','.join(sorted(ids)))).encode()).hexdigest()


def cand_seed(seed: str, epoch: int) -> str:
    """Candidate plan seed for the turn of epoch e: independent of the head's block seeds (no common random numbers)."""
    return hashlib.sha256(('%s:prior-baseline:cand:%d' % (seed, int(epoch))).encode()).hexdigest()


def plan_seed(seed: str, b: int) -> str:
    """Head block b plan seed (candidates use cand_seed: independent request seeds)."""
    return hashlib.sha256(('%s:prior-baseline:%d' % (seed, b)).encode()).hexdigest()


# ------------------------------------------------------------------ probe index mapping (one combined probe snapshot)
def probe_index(task_id: str, cfg=None) -> int:
    c = dict(DEFAULTS, **(cfg or {})); env, i = task_id.split(':')
    return int(i) + int(c['math_offset'] if env == 'math' else c['dapo_offset'])


def task_of(index: int, cfg=None) -> str:
    c = dict(DEFAULTS, **(cfg or {})); i = int(index)
    return ('dapo:%d' % (i - int(c['dapo_offset']))) if i >= int(c['dapo_offset']) else ('math:%d' % (i - int(c['math_offset'])))


# ------------------------------------------------------------------ estimate + decision
def _rate(cl):
    return cl['c'] / cl['n']


def estimate(drawn: dict, head: dict, cand: dict, env_size: dict, *, shares='size', tau=0.0, sigma_run_floor=0.0, bar_z=1.0, z_inclusive=False,
             cap_max=CAP_HIT_MAX, min_scored=100, min_cover=0.95) -> dict:
    """drawn = {task_id: dict(env, pi)}; head/cand = {task_id: probe cell {n, c, b, ...}}; env_size = {env: tasks in the pool}.
    -> dict(decision, reasons, evaluated, evaluated_private). evaluated holds no per-task data and no prior values."""
    P = sum(env_size.values())
    sh = {e: (n / P if shares == 'size' else 1.0 / len(env_size)) for e, n in env_size.items()}
    per = {}
    for t, dv in drawn.items():
        h, c = head.get(t), cand.get(t)
        if not (isinstance(h, dict) and isinstance(c, dict) and h.get('n') and c.get('n')): continue
        per.setdefault(dv['env'], []).append((t, float(dv['pi']), h, c))
    envs = {}; scored = 0; cb = {'h': [0.0, 0], 'c': [0.0, 0]}   # task-level cap-hit: capped / rollouts over every scored task
    for e, L in per.items():
        Ne = sum(1 / p for _, p, _, _ in L)
        def hj(f): return sum(f(h, c) / p for _, p, h, c in L) / Ne
        me = hj(lambda h, c: _rate(c) - _rate(h))
        Ve = 0.0
        for _, p, h, c in L:
            x, y = _rate(c), _rate(h); kc, kh = c['n'], h['n']
            v = x * (1 - x) / max(1, kc - 1) + y * (1 - y) / max(1, kh - 1); r = (x - y) - me
            Ve += ((1 - p) * r * r + p * v) / (p * p)
        envs[e] = dict(n=len(L), gain=me, var=Ve / (Ne * Ne), h_pass=hj(lambda h, c: _rate(h)), c_pass=hj(lambda h, c: _rate(c)),
                       h_cap=hj(lambda h, c: (h.get('b') or 0) / h['n']), c_cap=hj(lambda h, c: (c.get('b') or 0) / c['n']),
                       pq=hj(lambda h, c: _rate(h) * (1 - _rate(h)) * h['n'] / max(1, h['n'] - 1)))
        scored += len(L)
        for _, _, h, c in L:
            cb['h'][0] += h.get('b') or 0; cb['h'][1] += h['n']; cb['c'][0] += c.get('b') or 0; cb['c'][1] += c['n']
    cover = scored / max(1, len(drawn))
    have = [e for e in env_size if e in envs]; shn = sum(sh[e] for e in have) or 1.0
    shx = {e: sh[e] / shn for e in have}          # renormalised if an env has no scored task (fails min_cover anyway)
    delta = sum(shx[e] * envs[e]['gain'] for e in have) if have else None
    var_s = sum(shx[e] ** 2 * envs[e]['var'] for e in have)
    A = sum(shx[e] * envs[e]['pq'] for e in have); var_r = max(float(tau) ** 2 * 2 * A * A, float(sigma_run_floor) ** 2)
    se = math.sqrt(var_s + var_r) if have else None
    z = (delta / se) if (se and se > 0) else 0.0
    cap = cb['c'][0] / cb['c'][1] if cb['c'][1] else None      # candidate: capped rollouts / all rollouts on scored tasks
    hcap = cb['h'][0] / cb['h'][1] if cb['h'][1] else None     # head, same definition (recorded, not part of the rule)
    reasons = []
    if scored < min_scored: reasons.append('insufficient_scored_tasks(%d<%d)' % (scored, min_scored))
    if cover < min_cover: reasons.append('coverage %.3f < %.2f' % (cover, min_cover))
    if delta is None or not delta > 0: reasons.append('gain %s <= 0' % (None if delta is None else round(delta, 5)))
    if not ((z >= bar_z) if z_inclusive else (z > bar_z)): reasons.append(('z %.3f %s %.2f' if round(bar_z, 2) == bar_z else 'z %.3f %s %.4f') % (z, '<' if z_inclusive else '<=', bar_z))
    if cap is None or not cap < cap_max: reasons.append('cap_rate %s >= %.2f' % (None if cap is None else round(cap, 4), cap_max))
    ev = dict(z=z, gain=delta, delta=delta, se=se, se_task=math.sqrt(var_s), se_run=math.sqrt(var_r), tau_run=float(tau), sigma_run_floor=float(sigma_run_floor), n=scored,
              drawn=len(drawn), cover=cover, h_pass=sum(shx[e] * envs[e]['h_pass'] for e in have) if have else None,
              c_pass=sum(shx[e] * envs[e]['c_pass'] for e in have) if have else None, c_cap_rate=cap, h_cap_rate=hcap, cap_rule='task-level', cap_hit_max=cap_max,
              c_cap_hits=cb['c'][0], c_rollouts=cb['c'][1], h_cap_hits=cb['h'][0], h_rollouts=cb['h'][1],
              c_cap_rate_pool=sum(shx[e] * envs[e]['c_cap'] for e in have) if have else None,
              h_cap_rate_pool=sum(shx[e] * envs[e]['h_cap'] for e in have) if have else None,
              per_env={e: dict(n=envs[e]['n'], share=sh[e], gain=envs[e]['gain'], se=math.sqrt(envs[e]['var']),
                               h_pass=envs[e]['h_pass'], c_pass=envs[e]['c_pass'], c_cap=envs[e]['c_cap'], h_cap=envs[e]['h_cap'])
                       for e in have},
              estimator='per-env Hajek (inverse-probability ratio) estimate of Delta = sum_e share_e * mean_e(p_cand - p_head); '
                        'SE = sqrt(sampling + rollout variance + run term max(tau^2 * 2 * A^2, sigma_run_floor^2))', scoring=VERSION)
    return dict(decision='red' if reasons else 'green', reasons=reasons, evaluated=ev, z_min=bar_z, z_inclusive=bool(z_inclusive),
                cap_max=cap_max, min_tasks=min_scored, scoring=VERSION, sample_size=len(drawn), batch_overlap=None, batch_pre_post=None)


# ------------------------------------------------------------------ public-safety guard
PRIVATE_KEYS = ('"irt_ab"', '"theta"', '"a_used"', '"pi"', '"prior"', '"s_head"', '"n_head"', '"prior_s"', '"prior_n"', '"inclusion"', '"table_rows"', '"a0"', '"b0"')
OPEN_ROUND_KEYS = ('"secret"', '"seed"', '"ids"', '"indices"', '"per_task"', '"drawn_ids"', '"head_per_task"', '"candidate_per_task"')


def assert_public(doc, open_round: bool):
    s = json.dumps(doc, sort_keys=True)
    for k in PRIVATE_KEYS + (OPEN_ROUND_KEYS if open_round else ()):
        if k in s: raise RuntimeError('prior-v1 publish guard: %s in a public document%s' % (k, ' (round open)' if open_round else ''))
    for bad in ('r2.cloudflarestorage', 'X-Amz', 'aws_secret', 'secret_access', 'prior_table', '.sqlite'):
        if bad in s: raise RuntimeError('prior-v1 publish guard: %s' % bad)
    return True
