# z bar linear drop (from E514, 2026-11-01)

Jacob 2026-11-01 21:51 CR: replace the step-down (E513-E513, ops/zbar_stepdown) with a linear slow drop.

- **Rule:** GREEN iff gain > 0 AND z > z_bar AND cap-hit < 0.33 (task-level) AND >= 100 tasks scored, with
  **z_bar = max(0.25, 1.0 - 0.75 * h / 12)**, h = hours since the last GREEN: 1.00 right after a GREEN, 0.625 at 6 h, the 0.25
  floor from 12 h; back to 1.00 after every GREEN (`Z_BAR_SCHEDULE = {kind: linear, start: 1.0, floor: 0.25, hours_to_floor: 12}`
  in priors/baseline_v1.py; flag `heldout.prior_v1.z_bar_schedule`).
- **Clock** (unchanged from the step-down): h = (the candidate's TRAINING-RESULT time - decided_at of the last GREEN decision of an
  earlier epoch) / 3600, both public in https://affine.io/gate.json `entries[].timestamps`.
- **Published:** the exact z_bar used per decision (`z_bar` record, gate.json `entries[].thresholds.z_min`); gate.json `z_bar`
  (current, formula, floor_at, hours_until_floor), `/gate/zbar.json`, mining.json `heldout.z_bar`. The Gate z chart draws the
  per-turn bar as a sloped line ("z bar") with a dashed end to the bar now.
- Offline replay E497-E512 under the linear rule (with each candidate's real clock): no outcome changes (closest: E500-E502,
  z 0.75-0.79 against a bar of 0.83-0.86).
- Revert to the step-down only: `MODE=revert WHY=... bash activate_zlin.sh` (re-starts the step-down watchdog).
