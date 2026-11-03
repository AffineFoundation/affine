> **ARCHIVED 2026-11-03** (Jacob t1150u: "I think we should remove the multiplier it just adds complexity"). The king gain multiplier is
> removed: every payee is back to the equal 1/N split, the prior parent_hold is gone and mining.json no longer has
> `training.k1.king_gain_multiplier`. See `ops/gain_mult_removal/`. This folder is kept as history only.

# King gain multiplier (live from E581, 2026-11-02; Jacob pt1136u/pt1137u)

Each payee (king) crowned after the activation epoch (E580) gets weight (1/N) x m_i. The payees' shares are then
renormalized, so the total king mass is unchanged. Kings crowned at or before E580 (including 771eff20 at E567)
keep m = 1 permanently. There is no backfill.

- gain_i = mean over shared tasks of (king pass rate - parent pass rate). A shared task is one the prior sampler ran
  for both heads at the same token cap. Pass rates are pooled over every prior-sampler pass (k = 4).
  SE = sd(diff)/sqrt(n). cons_i = max(0, gain_i - 1 SE).
- Settled iff n_shared >= 2000. Until then m_i = 1.
- median = median of the last 10 positive cons values. The list is the frozen pre-activation full-pool analysis
  (E481 0.002952, E485 0.002721, E491 0.005021, E540 0.005578), followed by settled post-activation kings in
  crowning order. At activation the median is 0.003986. If it is undefined or <= 0, m = 1.
- m_i = clamp(1 + lambda (cons_i/median - 1), 0.5, 2), lambda = 1.
- Fail-safe: if the gate's prior cap no longer matches the last configured `cap_regimes` entry, the last state is
  kept frozen. Any read error in k1_service gives m = 1, which is the equal 1/N split. When every m is 1, the code
  takes the original `1.0/N` path, so the output is bit-identical.
- Parent coverage: `priors/controller.py` has an optional `parent_hold` in controller.json, enabled with 7000 tasks
  and max 1800 s. On a gate head change, if the served head (the new king's parent) has fewer than 7000 distinct
  tasks at the current cap, it keeps sampling that head for up to 30 min before switching. No new pods are used.

Files:
- `gain_mult.py`: run by the affine-gain-mult timer every 5 min. It writes `GAIN-MULT.private.json` (0600, gains/SE/n)
  and `GAIN-MULT.public.json` (per-king m rounded + settled only).
- `services/k1_service.py`: `gain_mults()` and the scores_doc share rule.
- `services/training_params.py`: mining.json `training.k1.king_gain_multiplier`.
- Patches: `patch_*.py`. Activation: `activate.sh`. Revert: `revert.sh`.

Dry-run proof (tests/dryrun-compare.txt): the hourly writer `--plan` gave an identical uids+weights vector before and after
activation (10 payees x 0.1, score_sha256 8f706971...). The backtest reproduces the 2026-11-02 full-pool analysis exactly.

Revert: `bash ~/subnet120-rewrite/state/root-audits/20261102-gain-mult/revert.sh`, then git revert this commit.
