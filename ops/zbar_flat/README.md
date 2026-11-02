# z bar fixed at 1.00 (from E553, 2026-11-02)

Jacob 2026-11-02 08:20 CR (pt1055u): remove the z bar reduction, keep it flat at 1 going forward.

- **Rule:** GREEN iff gain > 0 AND z > 1.00 (strict) AND cap-hit < 0.33 (task-level) AND >= 100 tasks scored.
- **Removed:** the time-relaxed bar (E513 steps, E514-E552 linear drop by hours since the last GREEN): `Z_BAR_SCHEDULE`,
  `z_bar_at`, `z_bar_next` (priors/baseline_v1.py), `last_green`, the clock and the schedule logic in `z_bar_for` /
  `z_bar_public` (priors/gate/prior_gate.py), the flag keys `heldout.prior_v1.z_bar_schedule / z_bar_clock / z_bar_from_epoch /
  z_bar_linear_from_epoch` (now `bar_z: 1.0`, `z_bar_fixed_from_epoch: 553`), the per-turn `turns` list in `/gate/zbar.json`,
  and the sloped per-turn bar line, its "z bar" legend and caption on the Gate z chart (now the flat dashed z = 1 bar, as before E513).
  The ops/zbar_stepdown and ops/zbar_linear packages are removed (history: a93d68c, a59734f).
- **Published:** every decision records `z_bar = {value: 1.0, kind: fixed}` and `thresholds.z_min = 1.0`; gate.json / mining.json
  `heldout.z_bar = {current: 1.0, kind: fixed, from_epoch: 553}`. The exact bar of E513-E552 stays in each entry's `thresholds.z_min`.
- Live after the e552 DECISION (activate_zflat.sh waits for a decision + a quiet gap, so no decision in flight changes).
  Watchdog `affine-zflat-watchdog` (watchdog_zflat.py) replaces `affine-zlin-watchdog` and auto-reverts on a bar != 1.0,
  gate errors, stalls or a missing cap rule.
- Revert (back to the linear drop): `MODE=revert WHY=... bash activate_zflat.sh` (validator host:
  ~/subnet120-rewrite/state/root-audits/20261102-zbar-flat/code/activate_zflat.sh, or the backup's revert.sh).
