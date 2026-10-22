"""SN120 v9 aggregation (CPU, pure functions; unit-tested in test_v9agg.py).  Rule from DESIGN-v9 / claims/miner_agg.py,
claims/adv_final.py:
  group score  s_g = 0.5 * sum_h mean_{k in half h, task(g) not in chunk k} (A_g . D_g[:,k]) / (8192 n)   (leave-task-out)
  unit         FIXED 3-group units per miner, unit = slot // 3 (slots 0-2, 3-5, 6-8). A partial unit (1-2 groups) is
               tested against the SAME 3-group null (conservative: fewer draws, never a gain). Splitting / regrouping gains nothing.
  null         epoch's out-of-sample honest-group null: 2-fold cross-fit by miner (fold = HS(seed,'fold',miner) % 2);
               a unit in fold f is calibrated on the other fold's groups. P_null(M >= m) = 1 - F_<(m)^3 (exact max-of-3 of
               the empirical group CDF); hit threshold q = P_null(M > med_M + 2 sd_M) (as in miner_agg.py).
  per miner    over the window: Stouffer Z = sum_b Phi^-1(1-p_b)/sqrt(B); binomial P(Binom(B, mean q) >= hits).
               gate = Z > Phi^-1(1 - a) AND binom < a, with a = alpha / N_UID (N_UID = scored UIDs in the window).
  payment      V = max(0, sum_g z_g) over the miner's groups for gated miners, z_g = (s_g - median_cal) / sd_cal; else 0."""
import collections, math
import numpy as np
from scipy.stats import norm, binom
UNIT = 3; NORM = 8192

def chunk_masks(chunks):
    halves = [np.array([c['kind'] == 'subject' and c['half'] == h for c in chunks]) for h in (0, 1)]
    tasks = [set(c['tasks']) for c in chunks]
    return halves, tasks

def group_score(rewards, dots, task, chunks, masks=None):
    halves, tasks = masks or chunk_masks(chunks)
    r = np.asarray(rewards, float); D = np.asarray(dots, float)
    if len(r) == 0 or D.shape != (len(r), len(chunks)): raise ValueError('dots shape')
    v = ((r - r.mean()) @ D) / (NORM * len(r))
    own = np.array([task in t for t in tasks]); parts = []
    for h in halves:
        m = h & ~own
        if m.any(): parts.append(v[m].mean())
    return float(np.mean(parts)) if parts else 0.0

class Null:
    """max-of-3 null from out-of-sample calibration group scores."""
    def __init__(self, cal, rng, nsim=200000):
        self.cal = np.sort(np.asarray(cal, float)); n = len(self.cal)
        if n < 30: raise ValueError('calibration too small (%d groups)' % n)
        M = self.cal[rng.integers(0, n, (nsim, UNIT))].max(1)
        self.med_M = float(np.median(M)); self.sd_M = float(M.std())
        self.q = self.p(self.med_M + 2 * self.sd_M, strict=True)
        self.med = float(np.median(self.cal)); self.sd = float(self.cal.std())
    def p(self, m, strict=False):
        n = len(self.cal); below = np.searchsorted(self.cal, m, side='right' if strict else 'left') / n
        below = min(max(below, 0.5 / n), (n - 0.5) / n)   # symmetric half-count continuity correction: p in (0, 1) so the
        return float(1.0 - below ** UNIT)                  # Stouffer term Phi^-1(1-p) is always finite (e85/e87 hit p=1 -> -inf)
    def z(self, s): return (s - self.med) / self.sd if self.sd > 0 else 0.0

def score_epoch(groups, chunks, fold_of, rng):
    """groups: list of dict(miner, uid, slot, task, rewards, dots). Returns per-miner units (aggregates only) + null info."""
    masks = chunk_masks(chunks); G = []
    for g in groups:
        if any(v is None for v in g['rewards']): continue           # invalid group: not scored, not in the null
        G.append(dict(g, s=group_score(g['rewards'], g['dots'], g['task'], chunks, masks)))
    nulls = {}
    for f in (0, 1):
        nulls[f] = Null([g['s'] for g in G if fold_of(g['miner']) != f], rng)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for g in G: by[g['miner']][g['slot'] // UNIT].append(g)
    miners = {}
    for m, units in sorted(by.items()):
        nl = nulls[fold_of(m)]; rows = []
        for u, gs in sorted(units.items()):
            M = max(g['s'] for g in gs); p = nl.p(M)
            rows.append(dict(unit=u, groups=len(gs), p=p, q=nl.q, hit=bool(p < nl.q), sum_z=float(sum(nl.z(g['s']) for g in gs))))
        uids = sorted({g['uid'] for us in units.values() for g in us if g['uid'] is not None})
        miners[m] = dict(uid=uids[0] if len(uids) == 1 else None, units=rows)
    info = {str(f): dict(cal_groups=len(nl.cal), q=nl.q, med=nl.med, sd=nl.sd, med_M=nl.med_M, sd_M=nl.sd_M) for f, nl in nulls.items()}
    return miners, info, len(G)

def gate_window(per_epoch, alpha=0.01):
    """per_epoch: list of score_epoch miner dicts (one per epoch in the window). Stouffer + binomial, Bonferroni over scored UIDs."""
    acc = collections.defaultdict(lambda: dict(p=[], q=[], hits=0, sum_z=0.0, uid=None))
    for miners in per_epoch:
        for m, r in miners.items():
            a = acc[m]; a['uid'] = r['uid'] if r['uid'] is not None else a['uid']
            for u in r['units']:
                a['p'].append(u['p']); a['q'].append(u['q']); a['hits'] += u['hit']; a['sum_z'] += u['sum_z']
    N = max(1, len(acc)); ab = alpha / N; out = {}
    for m, a in sorted(acc.items()):
        B = len(a['p']); qm = float(np.mean(a['q'])); Z = float(norm.isf(np.array(a['p'])).sum() / math.sqrt(B))
        pb = float(binom.sf(a['hits'] - 1, B, qm))
        gated = bool(Z > norm.isf(ab) and pb < ab)
        out[m] = dict(uid=a['uid'], B=B, hits=int(a['hits']), stouffer_z=Z, binom_p=pb, gated=gated,
                      flag_stouffer_uncorrected=bool(Z > norm.isf(alpha)), flag_binom_uncorrected=bool(pb < alpha),
                      V=float(max(0.0, a['sum_z'])) if gated else 0.0)
    summ = dict(scored_uids=len(acc), alpha=alpha, alpha_per_uid=ab, gated=sum(r['gated'] for r in out.values()),
                flag_rate_stouffer_uncorrected=float(np.mean([r['flag_stouffer_uncorrected'] for r in out.values()])) if out else None,
                flag_rate_binom_uncorrected=float(np.mean([r['flag_binom_uncorrected'] for r in out.values()])) if out else None,
                mean_batches=float(np.mean([r['B'] for r in out.values()])) if out else None)
    return out, summ
