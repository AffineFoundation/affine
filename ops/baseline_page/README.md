# affine.io/baseline (Jacob, 2026-11-02)

Public page with the SHAPE of the live prior-v1 baseline (gate pool gate-pool-v1, 19,305 tasks; prior-v1 live since E497):

1. Head pass rate by task: tasks sorted by s/n from the live prior table (ascending; ties by the posterior mean (s+1)/(n+2)).
2. Selection probability per round: the per-task inclusion probability pi of the Neyman/Hajek draw
   (`baseline_v1.inclusion`, N / floor / cap / shares / n_init of the current round), in the same task order.

Privacy: only binned shapes are published. `site-tools/build_baseline_curves.py` averages each curve over 160 equal-count bins
(~120 tasks each) and rounds (pass rate 2 decimals; pi in % with 2 decimals). No task ids, no per-task values, no table digest;
a guard refuses to write output containing task-id prefixes or long arrays. The prior table is opened read-only.

Deploy (AffineValidator):
- `website/baseline/{index.html,baseline.css,baseline.js}` (this dir's `site/baseline/`), data `website/baseline/curves.json`.
- `site-tools/build_baseline_curves.py`, run by the user timer `affine-site-baseline.timer` every 15 min (writes only on change).
- Nav link "Baseline" in `website/index.html`; Caddy (affine.io block only) `rewrite /baseline /baseline/index.html`. See changes.diff.

Revert: `~/subnet120-rewrite/state/bench-results/site-backup/20261102T142823Z/revert.sh` on AffineValidator (copy: revert.sh).

## v2 (Jacob pt1059u, 2026-11-02)
- Caption fixed: the draw follows the Neyman allocation, weight = sqrt of v_i (not p(1-p)).
- New "How the draw works" section: the inclusion formula as in `priors/baseline_v1.py` `inclusion()`
  (v_i = E[p(1-p)] under Beta(s+1, n-s+1) = AB/((A+B)(A+B+1)), as the code computes it), cap + redistribution,
  commit-reveal Poisson draw, the knobs with live values (curves.json now carries cfg: N, floor, cap, n_init, shares),
  and the per-env Hajek estimator. Plain HTML/CSS math, no external library. No new data published.
- Revert: `~/subnet120-rewrite/state/bench-results/site-backup/20261102T151156Z/revert.sh` (copy: revert-v2.sh).
