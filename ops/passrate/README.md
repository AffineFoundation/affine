# affine.io first tile: pass rate across all envs (Jacob pt1130u, 2026-11-02)

"Remove this and replace it just with the general pass rate across our envs. Measured from the prior. Because we are planning
to scale well beyond just our math problems." The Performance tile (affine_math full set + private holdout, performance.json)
is replaced by one series: each king's pass rate over the whole gate pool (gate-pool-v1, affine_math + dapo, 19,305 tasks).

## Data (passrate.json, aggregates only)
Per king (x = crowned epoch), best source first:
1. `prior-pass`: the prior sampler pod runs the current head on every pool task, k=4 rollouts per task per pass. Only COMPLETE
   passes count (a pass cut short by a head change keeps the tasks that finished first - shorter, easier - and reads 1-20 points
   high). Pass rate = mean over all 19,305 tasks of correct/graded over the head's complete passes at its latest max_tokens.
   SE = rollout noise only (census): sqrt(sum p(1-p)/(n-1)) / N, about 0.05-0.1 points.
2. `gate-ipw` (until a king's first complete pass, ~1 h after crowning, or when the prior pod was down): the gate's head baseline
   of each prior-v1 round (~500 tasks, inclusion probabilities pi, 12 rollouts) re-weighted by 1/pi: per-env Hajek, combined by
   env size, the gate's own variance formula; rounds of one king at one cap combined by inverse variance. 95% CI about +-3 points.
Cross-check where both exist: 3eceb890 76.1 (prior) vs 74.3-77.1 (4 gate rounds); 6bfabf5d 76.0 vs 76.1/76.6; 100802ca 79.0 vs
80.0 (8K)/78.8 (12K); 771eff20 81.9 vs 81.6. Same king 100802ca at 8K vs 12K on the gate draw: no visible shift -> no cap marker.
Kings without a complete prior pass and without a prior-v1 gate round (8ac947ac, 1d271188, cb92bdc5) are not shown.

Private, never published: prior table values, per-task grades, task ids of draws, inclusion probabilities, IRT parameters, holdout.
The builder has a publish guard (no pi/ids/per_task/task-id strings).

## Pieces
- `build_passrate.py` -> host `~/subnet120/affine/site-tools/`, systemd --user `affine-site-passrate.timer` (5 min). Private
  incremental cache of per-(head, pass) sums: RA/20261102-passrate/state (0700).
- `site/perf.js` -> website/perf.js (same tile id, frame, colors and load sequence as before; ?v=pt1130u-passrate-20261102).
- `patch_index.py`, `patch_llms.py`: in-place edits of the live index.html (tile markup, prefetch list, ?v=) and llms.txt (section
  "Pass rate across all envs", endpoint list).
- `ops/performance_eval/perf_service.py`: the private eval keeps running; its full doc now goes to
  RA/20261031-performance-eval/state/performance.private.json (0600) and the public performance.json keeps only the king list
  (bench daemon input: king, epoch, era, parent, crowned_at).

## Deploy / revert (host)
- `bash RA/20261102-passrate/code/deploy.sh` (backs up to RA/20261102-passrate/backups/<ts>, LAST-BACKUP).
- Revert: `bash ~/subnet120-rewrite/state/root-audits/20261102-passrate/code/revert.sh` (refuses if index.html/llms.txt changed
  since the deploy unless FORCE=1), then `git revert` of this commit.
