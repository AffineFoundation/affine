# irt-info: IRT information-weighted baseline draw (live from 2026-11-02 10:31 CR)

The gate's per-round baseline draw (prior-v1 rounds, `priors/gate/prior_gate.py`) can weight tasks two ways, set by
`heldout.prior_v1.weighting` in the gate flag and frozen into each round when it opens:

| weighting | draw score per task | live |
|---|---|---|
| `prior-v1` | sqrt(E[p(1-p)]) under Beta(s+1, n-s+1) from the prior table | E497 to 2026-11-02 |
| `irt-info` | sqrt(I), I = a'^2 P(1-P) at the head's ability (2PL Fisher information) | from 2026-11-02 (Jacob Steeves) |

Everything else is unchanged: pi_i = N [(1 - floor) g_i / sum g + floor w_i / sum w], g_i = w_i * score_i, N = 500, floor 0.10,
cap 0.10 (excess re-spread), Poisson draw with the commit-reveal seed, per-env Hajek estimate, and the GREEN rule
(gain > 0, z > 1 flat, pooled cap-hit < 0.33, >= 100 scored). The estimate is unbiased for any weights; the weighting only changes
where the ~500 tasks go (and so the SE).

## Model
- 2PL item response: P_i(theta) = 1 / (1 + exp(-a_i (theta - b_i))); a_i discrimination, b_i difficulty, theta checkpoint ability
  (standardised: mean 0, sd 1 over the fitted checkpoints).
- Fit (`fit.py ab 1.0 4 chain`): MAP on every recorded checkpoint x gate-pool task pass/fail (`gather.py`: prior-sampler cells,
  held-out head blocks, epoch head/candidate probes, the A/A noise runs; offline arms excluded). Priors: log a_i ~ N(env mean, 1),
  b_i ~ N(env mean, 4^2). First fit: 365 checkpoints, 19,305 tasks, 3.06M rollouts.
- Daily refit (`refit.sh`, user timer `affine-irt-refit` 03:30 CR, nice 19 / ionice idle / 4 threads, waits while load >= 16);
  `export_params.py` replaces the live params only if all pool tasks are present, values are finite, the median a is sane and the
  rank correlation of a with the previous params is > 0.5.
- Head ability theta_h: `baseline_v1.theta_map`, 1-D MAP (N(0, 2^2) prior) of the head's own pass/fail on gate-pool tasks
  (its winning candidate probe, its head blocks, prior-table cells of that head), recomputed at every round open. Fallbacks:
  the fit's theta for that checkpoint, then the previous irt round's theta, then the top fitted checkpoint.
- a is weakly determined for many tasks (most checkpoints come from one training chain), so the draw uses
  a'_i = exp(0.5 log a_i + 0.5 log m) clipped to [0.5 m, 2 m], m = median a. P_i itself uses the fitted a_i, b_i.

## Privacy
Per-task a_i, b_i, the head's theta_h and the inclusion probabilities pi_i are private (like the prior table). Each round publishes
the sha256 of the params file it used (`irt.params_sha256` in /gate/heldout/current.json and the round reveal), the weighting and
the knobs (settings.weighting, a_shrink, a_lo, a_hi, theta_prior_sd).

## Operations
- `activate_irt.sh` (install in an idle gap; MODE=revert WHY=... to go back to prior-v1 weighting: flag off, closes an open irt
  round so a prior-v1 round re-opens at the next tick, code / llms.txt / baseline page restored where unchanged).
- `watchdog_irt.py` (unit affine-irt-watchdog, 72 h): auto-revert on an IRT fallback at round open, a draw size off N, a draw not
  done 30 min after the deadline, a stalled head baseline (30 min), a fail-closed turn, or >= 3 gate errors in 20 min.
- If the IRT weights cannot be computed at a round open, that round uses prior-v1 weighting (event `prior_irt_fallback`).

## 2026-11-02 stagger
The 12K max_tokens switch (Jacob pt1084u) was set to land at the same round open. To keep the two effects attributable, the
weighting was switched off at 11:14 CR (`stagger_arm.sh`): the next round opens with 12K + prior-v1 weighting, and once it has
opened the script sets weighting = irt-info again, so the round after it is the first irt-info round.
