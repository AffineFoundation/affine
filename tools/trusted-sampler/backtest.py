#!/usr/bin/env python3
"""Backtest: per-epoch paired deltas -> per-miner (UID) credit regression.

inputs: pooled.json (aggregate.py --json), inclusion.json (extract_inclusion.py), current_pay.json
model : delta_e = tau0 + sum_i z_ie * u_i + eps_e,  eps_e ~ N(0, se_e^2),  u_i ~ N(0, s_u^2)
        z_ie = share of epoch e's trained groups that came from UID i (shares sum to 1, so a common
        per-miner level tau0 is absorbed by the intercept; only deviations u_i are identifiable).
        tau_i = tau0 + u_i. s_u is set by empirical Bayes (marginal likelihood grid) and also
        reported at fixed plausible values.
"""
import json, math, sys
import numpy as np

pooled = json.load(open(sys.argv[1])); inc = json.load(open(sys.argv[2])); pay = json.load(open(sys.argv[3]))
out_json = sys.argv[4] if len(sys.argv) > 4 else None
by_out = {r['output_checkpoint'][:8]: r for r in inc if r.get('output_checkpoint')}
rows = []
for p in pooled:
    r = by_out.get(p['after'])
    if r and r['input_checkpoint'][:8] == p['before']:
        rows.append((r, p))
E = len(rows)
uids = sorted({u for r, _ in rows for u in r['trained_by_uid'] if u != 'None'}, key=int)
U = len(uids); ui = {u: k for k, u in enumerate(uids)}
Z = np.zeros((E, U)); Zi = np.zeros((E, U))
for e, (r, p) in enumerate(rows):
    tot = sum(r['trained_by_uid'].values())
    for u, n in r['trained_by_uid'].items():
        if u in ui:
            Z[e, ui[u]] = n / tot; Zi[e, ui[u]] = 1
d = np.array([p['delta'] for _, p in rows]); s = np.array([p['se_paired'] for _, p in rows])


def fit(Zm, d, s, su):
    """GLS intercept + ridge posterior for u. Returns tau0, se_tau0, post mean u, post sd u, loglik."""
    W = np.diag(1 / s ** 2)
    C = np.diag(s ** 2) + su ** 2 * Zm @ Zm.T            # marginal cov of d given tau0
    Ci = np.linalg.inv(C); one = np.ones(len(d))
    tau0 = (one @ Ci @ d) / (one @ Ci @ one); se0 = math.sqrt(1 / (one @ Ci @ one))
    r = d - tau0
    _, logdet = np.linalg.slogdet(C)
    ll = -0.5 * (r @ Ci @ r + logdet + len(d) * math.log(2 * math.pi))
    P = Zm.T @ W @ Zm + np.eye(Zm.shape[1]) / su ** 2      # posterior precision of u
    Pi = np.linalg.inv(P); mu = Pi @ Zm.T @ W @ r
    return tau0, se0, mu, np.sqrt(np.diag(Pi)), ll


grid = [10 ** x for x in np.linspace(-4, 1, 51)]
lls = [fit(Z, d, s, g)[4] for g in grid]
su_eb = grid[int(np.argmax(lls))]
res = {}
for label, su in (('eb', su_eb), ('su=0.01', 0.01), ('su=0.05', 0.05), ('su=0.25', 0.25)):
    tau0, se0, mu, sd, ll = fit(Z, d, s, su)
    z = mu / sd
    res[label] = dict(s_u=su, tau0=tau0, se_tau0=se0, n_distinguishable_2sd=int((np.abs(z) > 2).sum()),
                      max_abs_z=float(np.abs(z).max()), post_sd_over_prior_median=float(np.median(sd / su)), loglik=ll)
# split-half (alternate epochs), at a fixed weak prior so the fit is data-driven
def split_corr(su):
    A = [e for e in range(E) if e % 2 == 0]; B = [e for e in range(E) if e % 2 == 1]
    if len(A) < 1 or len(B) < 1: return None
    ma = fit(Z[A], d[A], s[A], su)[2]; mb = fit(Z[B], d[B], s[B], su)[2]
    both = (Z[A].sum(0) > 0) & (Z[B].sum(0) > 0)
    if both.sum() < 3: return None
    from scipy.stats import spearmanr, pearsonr
    return dict(miners_in_both=int(both.sum()), pearson=float(pearsonr(ma[both], mb[both])[0]),
                spearman=float(spearmanr(ma[both], mb[both])[0]))
split = {str(su): split_corr(su) for su in (0.05, 0.25, 1.0)}
# would-be pay from posterior tau (EB prior), vs current pay
tau0, se0, mu, sd, _ = fit(Z, d, s, 0.25)
tau = tau0 + mu
cur = pay['weights_by_uid']
common = [u for u in uids if u in cur]
wb = np.maximum(np.array([tau[ui[u]] for u in common]), 0); wb = wb / wb.sum() if wb.sum() > 0 else np.ones(len(common)) / len(common)
wc = np.array([cur[u] for u in common]); wc = wc / wc.sum()
from scipy.stats import spearmanr
def gini(w):
    w = np.sort(w); n = len(w); return float((2 * np.arange(1, n + 1) - n - 1) @ w / (n * w.sum()))
part_rate = Zi.mean(0)
pay_cmp = dict(uids_compared=len(common), spearman=float(spearmanr(wb, wc)[0]), tv_distance=float(0.5 * np.abs(wb - wc).sum()),
               gini_would_be=gini(wb), gini_current=gini(wc), top10_overlap=len(set(np.argsort(-wb)[:10]) & set(np.argsort(-wc)[:10])),
               corr_would_be_vs_share=float(spearmanr(wb, Z.sum(0)[[ui[u] for u in common]])[0]),
               corr_current_vs_share=float(spearmanr(wc, Z.sum(0)[[ui[u] for u in common]])[0]))
# power: Monte Carlo with the real design. true u ~ N(0, su_true); how often is a miner's sign recovered / |z|>2?
rng = np.random.default_rng(0)
pbar = Z.mean(0); pbar = pbar / pbar.sum()
def power(E_new, se, su_true, reps=8):
    """Fresh simulated epochs: 256 groups multinomially drawn with each UID's observed mean share."""
    hits, cors = [], []
    for _ in range(reps):
        Zr = rng.multinomial(256, pbar, size=E_new) / 256.0; sr = np.full(E_new, se)
        u = rng.normal(0, su_true, U); dd = 0.002 + Zr @ u + rng.normal(0, sr)
        _, _, m, sdv, _ = fit(Zr, dd, sr, su_true)
        hits.append(float((np.abs(m / sdv) > 2).mean())); cors.append(float(np.corrcoef(m, u)[0, 1]))
    return dict(epochs=E_new, se_per_epoch=se, true_sd_u=su_true, frac_miners_z_gt_2=float(np.mean(hits)),
                corr_true_vs_est=float(np.mean(cors)))
pw = [power(E_, se_, su_) for su_ in (0.01, 0.05) for E_, se_ in ((6, float(s.mean())), (100, float(s.mean())), (1000, float(s.mean())), (1000, 0.001))]
summary = dict(epochs=[dict(epoch=r['epoch'], step=r['optimizer_step'], before=p['before'], after=p['after'], delta=p['delta'],
                            se_paired=p['se_paired'], se_unpaired=p['se_unpaired'], tasks=p['tasks'], records=p['records'],
                            uids_trained=sum(1 for u in r['trained_by_uid'] if u != 'None'), groups=r['trained_total']) for r, p in rows],
               n_epochs=E, n_uids=U, mean_uids_per_epoch=float(Zi.sum(1).mean()), mean_epochs_per_uid=float(Zi.sum(0).mean()),
               share_stats=dict(mean_nonzero=float(Z[Z > 0].mean()), max=float(Z.max())),
               delta_mean=float(np.average(d, weights=1 / s ** 2)), delta_between_epoch_sd=float(d.std(ddof=1)) if E > 1 else None,
               se_paired_mean=float(s.mean()), fits=res, split_half=split, pay=pay_cmp, power=pw)
print(json.dumps(summary, indent=1))
if out_json: json.dump(summary, open(out_json, 'w'), indent=1)
