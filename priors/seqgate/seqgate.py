"""SN120 anytime-valid sequential gate (SHADOW ONLY, Jacob Steeves approved 2026-11-02 11:14 CR).

Nothing here is imported by the live gate. Pure functions, no I/O, no private data.

Model
-----
Pool P (gate pool, 19,305 tasks), per-task pool weight w_i = share_e / P_e (size shares -> w_i = 1/P, same as live).
Baseline draw: the live inclusion rule (prior-v1 or irt-info pi_i with sum pi = N) at N = 2,000 (cap raised, see below).
Head: 8 seeded rollouts/task on every drawn task, cached once per head (pass rate + cap-hits per task).
Candidate: the drawn tasks are visited in a uniformly random (committed) order, in batches of ~100.
Per visited task: d_i = p_cand,i - p_head,i in [-1, 1], importance weight r_i = w_i / q_i, q_i = pi_i / N (Horvitz-Thompson,
with-replacement form), X_i = r_i d_i.  E[X] = sum_i w_i E[d_i] = Delta (the pool-weighted mean gain, the live estimand).
|X_i| <= B = max_i r_i over the drawn set (fixed before the candidate's first task -> predictable).

Test (Waudby-Smith & Ramdas 2023, JRSS-B, "Estimating means of bounded random variables by betting", predictable
plug-in betting capital with empirical-Bernstein-style lambda; default lam_mode='mix' = average of sample-size-tuned capitals,
see Bettor):
  GREEN  H0: Delta <= 0       K+_t(m) = prod_j (1 + l_j (X_j - m)),  l_j = min(lam*_j, c / (m - lo))  -> GREEN when K+_t(0) >= 1/alpha
  RED    H0': Delta >= delta  K-_t(m) = prod_j (1 - l_j (X_j - m)),  l_j = min(lam*_j, c / (hi - m)) -> RED when K-_t(delta) >= 1/alpha_red
  lam*_j: 'prpl' sqrt(2 log(1/alpha) / (s2_{j-1} j log(1 + j))); 'tuned' sqrt(2 log(1/alpha) / (s2_{j-1} n*)); 'mix' (default) =
  average of the 'tuned' capitals over n* in tune_ns (predictable: uses X_1..X_{j-1} only)
  s2_j   = (s2_0 + sum_{k<=j} (X_k - mu_{k-1})^2) / (j + 1),  mu_j = (mu_0 + sum_{k<=j} X_k)/(j + 1)
Both K are nonnegative supermartingales under their nulls, so by Ville P(ever GREEN | Delta <= 0) <= alpha and
P(ever early-RED | Delta >= delta) <= alpha_red, whatever the batch schedule. LCB/UCB = inversion of K+/K- over m (bisection).
Budget exhausted without GREEN -> RED. The pooled cap-hit rule (>= 33% of candidate rollouts hit the task cap -> RED) is a separate
hard check, exactly as live. Seeds: independent (head plan seed != candidate plan seed) unless paired_seeds=True (CRN switch).

Caveat (documented in README): the drawn set is a Poisson sample visited without replacement; we treat visits as iid draws from q.
With N/P ~ 10% the finite-population effect only shrinks the variance (conservative), and rollout noise is independent per task.
Run-to-run offsets (sigma_run) are NOT covered: a shared offset between the head run and the candidate run is a real mean shift for the
test. A/A measured sigma_run ~ 0 (95% ub ~0.003-0.008); green_margin (default 0) is the knob if that changes.
"""
import hashlib
import math
import random

DEFAULTS = dict(alpha=0.03, alpha_red=0.03, delta=0.002, budget=2000, batch=100, c=0.75, s2_0=0.01, mu_0=0.0, lam_mode='mix',
                tune_ns=(250, 500, 1000, 2000), bounds=True,
                green_margin=0.0, cap_max=0.33, min_scored=100, paired_seeds=False)


def order_seed(round_seed: str, epoch: int) -> str:
    return hashlib.sha256(('seqgate-shadow-v1:order:%s:%d' % (round_seed, int(epoch))).encode()).hexdigest()


def visit_order(task_ids, seed: str):
    L = sorted(task_ids); rnd = random.Random(int(seed[:16], 16)); rnd.shuffle(L); return L


def ht_terms(order, head, cand, pi, w, N):
    """-> list of (task, X_i, d_i, r_i). head/cand = {task: {n, c, b}}; pi = {task: pi_i}; w = {task: w_i}; N = sum pool pi."""
    out = []
    for t in order:
        h, c = head.get(t), cand.get(t)
        if not (h and c and h.get('n') and c.get('n')): continue
        d = c['c'] / c['n'] - h['c'] / h['n']; r = w[t] * N / pi[t]
        out.append((t, r * d, d, r))
    return out


class Bettor:
    """One-sided betting capital for H0: mean <= m (side=+1, GREEN) or mean >= m (side=-1, RED); X in [lo, hi].
    lam_mode: 'prpl'  = WSR (2023) predictable plug-in lam_t = sqrt(2 log(1/a) / (s2_{t-1} t log(1+t)))  (anytime-tuned)
              'tuned' = lam_t = sqrt(2 log(1/a) / (s2_{t-1} n*))  (WSR's sample-size-tuned plug-in, n* = budget)
              'mix'   = equal-weight average of the capitals of 'tuned' at every n* in tune_ns (an average of
                        test supermartingales is one, so Ville still gives P(sup K >= 1/a) <= a).
    Every lam_t uses X_1..X_{t-1} only (predictable); truncation lam <= c/(m - lo) (GREEN) or c/(hi - m) (RED), c < 1."""

    def __init__(self, lo, hi, alpha, c=0.75, s2_0=0.01, mu_0=0.0, lam_mode='mix', tune_ns=(250, 500, 1000, 2000)):
        self.lo, self.hi, self.alpha, self.c = lo, hi, alpha, c
        self.s2_0 = s2_0; self.s2_sum = s2_0; self.mu_sum = mu_0; self.t = 0; self.xs = []; self.s2s = []
        self.ns = [None] if lam_mode == 'prpl' else ([tune_ns[-1]] if lam_mode == 'tuned' else list(tune_ns))

    def add(self, x):
        t = self.t + 1
        self.s2s.append(max(self.s2_sum / t, 1e-8)); self.xs.append(x)   # s2_{t-1} = (s2_0 + sum_{k<t}(X_k - mu_{k-1})^2) / t
        mu_prev = self.mu_sum / t
        self.s2_sum += (x - mu_prev) ** 2; self.mu_sum += x; self.t = t

    def _lam(self, t, s2, n):
        L = math.log(1 / self.alpha)
        return math.sqrt(2 * L / (s2 * t * math.log(1 + t))) if n is None else math.sqrt(2 * L / (s2 * n))

    def logK(self, m, side):
        cap = self.c / max((m - self.lo) if side > 0 else (self.hi - m), 1e-12)
        logs = []
        for n in self.ns:
            s = 0.0
            for i, (x, s2) in enumerate(zip(self.xs, self.s2s)):
                l = min(self._lam(i + 1, s2, n), cap); s += math.log1p(side * l * (x - m))
            logs.append(s)
        mx = max(logs)
        return mx + math.log(sum(math.exp(v - mx) for v in logs) / len(logs))

    def bound(self, side, tol=1e-5):
        """side=+1: LCB = inf{m: K+(m) < 1/alpha}; side=-1: UCB = sup{m: K-(m) < 1/alpha}."""
        thr = math.log(1 / self.alpha); lo, hi = self.lo, self.hi
        if side > 0:
            if self.logK(lo + 1e-9, +1) < thr: return lo
            a, b = lo + 1e-9, hi - 1e-9
            while b - a > tol:
                m = (a + b) / 2
                if self.logK(m, +1) >= thr: a = m
                else: b = m
            return a
        if self.logK(hi - 1e-9, -1) < thr: return hi
        a, b = lo + 1e-9, hi - 1e-9
        while b - a > tol:
            m = (a + b) / 2
            if self.logK(m, -1) >= thr: b = m
            else: a = m
        return b


def run_sequential(terms, cap_stats=None, cfg=None, final=True):
    """terms = ht_terms(...) in visit order (already truncated to what exists). Batched decision.
    cap_stats = list of (cand_capped, cand_rollouts) aligned with terms (pooled cap rule, checked at every batch end on what was seen).
    final=False: terms is a prefix of a longer committed order still being generated; no decision yet -> decision None.
    -> dict(decision, reason, tasks_used, batches, trace=[per-batch dict])."""
    c = dict(DEFAULTS, **(cfg or {}))
    budget = min(int(c['budget']), len(terms)); bs = int(c['batch'])
    if budget == 0: return dict(decision='red', reason='no scored tasks', tasks_used=0, batches=0, trace=[], truncated=True)
    B = float(c['B']) if c.get('B') else max(r for _, _, _, r in terms[:budget]); lo, hi = -B, B   # B: max r_i over the whole drawn set (fixed up front)
    tn = tuple(int(x) for x in c['tune_ns'])
    gp = Bettor(lo, hi, c['alpha'], c['c'], c['s2_0'], c['mu_0'], c['lam_mode'], tn)
    rd = Bettor(lo, hi, c['alpha_red'], c['c'], c['s2_0'], c['mu_0'], c['lam_mode'], tn)
    trace = []; cs = [0, 0]; i = 0; nb = 0
    if not final: budget = min(budget, len(terms) // bs * bs)   # only whole batches of a prefix
    if budget == 0: return dict(decision=None, reason='waiting', tasks_used=0, batches=0, trace=[])
    while i < budget:
        j = min(budget, i + bs)
        for k in range(i, j):
            gp.add(terms[k][1]); rd.add(terms[k][1])
            if cap_stats: cs[0] += cap_stats[k][0]; cs[1] += cap_stats[k][1]
        i = j; nb += 1
        lgG = gp.logK(float(c['green_margin']), +1); lgR = rd.logK(float(c['delta']), -1)
        n = i; mean = sum(x for _, x, _, _ in terms[:n]) / n
        cap = cs[0] / cs[1] if cs[1] else None
        row = dict(tasks=n, mean_ht=mean, e_green=math.exp(min(lgG, 700)), e_red=math.exp(min(lgR, 700)),
                   lcb=gp.bound(+1) if c['bounds'] else None, ucb=rd.bound(-1) if c['bounds'] else None, cap=cap)
        trace.append(row)
        if n >= int(c['min_scored']) and lgG >= math.log(1 / c['alpha']):
            if cap is not None and cap >= c['cap_max']:   # pooled cap-hit rule: separate hard check, blocks the GREEN (live: RED)
                return dict(decision='red', reason='cap_rate %.4f >= %.2f (e_green %.1f crossed)' % (cap, c['cap_max'], row['e_green']), tasks_used=n,
                            batches=nb, trace=trace, truncated=budget < int(c['budget']), cap_red=True)
            return dict(decision='green', reason='e_green %.1f >= %.1f (lcb %s > %.4f)' % (row['e_green'], 1 / c['alpha'], row['lcb'] if row['lcb'] is None else round(row['lcb'], 5), c['green_margin']),
                        tasks_used=n, batches=nb, trace=trace, truncated=budget < int(c['budget']))
        if lgR >= math.log(1 / c['alpha_red']):
            return dict(decision='red', reason='early: e_red %.1f >= %.1f (ucb %s < delta %.4f)' % (row['e_red'], 1 / c['alpha_red'], row['ucb'] if row['ucb'] is None else round(row['ucb'], 5), c['delta']),
                        tasks_used=n, batches=nb, trace=trace, truncated=budget < int(c['budget']), early=True)
    if not final: return dict(decision=None, reason='continue', tasks_used=budget, batches=nb, trace=trace)
    return dict(decision='red', reason='budget %d exhausted (lcb %s)' % (budget, trace[-1]['lcb']), tasks_used=budget, batches=nb, trace=trace,
                truncated=budget < int(c['budget']))
