# z bar step-down (from E513, 2026-11-01)

Jacob 2026-11-01 21:23 CR: "Go with the step-down: z>1, 0.5 after 6h, 0.25 after 12h without a GREEN, reset on GREEN".

- **Rule:** GREEN iff gain > 0 AND z > z_bar AND cap-hit < 0.33 (task-level) AND >= 100 tasks scored. z_bar = 1.00 below 6 h,
  0.50 from 6 h, 0.25 from 12 h since the last GREEN; back to 1.00 after every GREEN (`Z_BAR_SCHEDULE` in priors/baseline_v1.py,
  flag `heldout.prior_v1.z_bar_schedule` / `z_bar_clock = since_last_green`).
- **Clock (deterministic, public):** hours = (the candidate's TRAINING-RESULT time - decided_at of the last GREEN decision of an
  earlier epoch) / 3600. Both are in https://affine.io/gate.json (`entries[].timestamps.training_result`, `.decided_at`); the
  training-result time is used (not the decision time) so gate latency cannot move the bar.
- **Published:** each decision records `z_bar` (value, hours_since_green, last_green_epoch, schedule, next_step) and uses it as
  `z_min`/`bar_z`; gate.json `z_bar` (current bar + next step), `/gate/zbar.json` (also the per-turn bar), mining.json
  `heldout.z_bar`. The affine.io Gate z chart draws the per-turn bar as a step line ("z bar"), extended to the current/next value.
- Draw, grader, rollouts, the cap rule and everything else are unchanged; head and candidates are graded identically.
- Offline replay E497-E511: no outcome changes (every candidate's clock was < 6 h after the E491 GREEN, so z_bar = 1.00).
- Files: `activate_zbar.sh` (install at a quiet gap, health check, `MODE=revert` reverts only this change), `watchdog_zbar.py`,
  `setflag_zbar.py`, `patch_texts_zbar.py`, `replay_zbar.py`, `site/k1.js`.
