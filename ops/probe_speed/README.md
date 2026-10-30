# SN120 probe-speed + gate fail-safe changes (2026-10-30, live on the validator / probe pod)

Live copies (sha256 prefix = what runs):
- `ops/promotion_gate/gate_service.py` (67679d75): a candidate the gate will not judge (held-out round closed, candidate input != sampled
  head) is decided **RED** (`skip_red`, event `gate_not_judged_red`) instead of `skip`; the v829i lifecycle accepts only green/red and
  turned `skip` into a timeout-promote (fail open). Also `delta_after_result` (EPOCH-TIME opt-in): an early-staged delta make waits for
  TRAINING-RESULT (early makes contended with the busy trainer: 447 s vs 53 s). Revert: `MODE=revert bash activate_skipred.sh`.
- `ops/probe_pod/podctl.py` (511142f1, ps1): the 8 vLLM replicas start in parallel (per-replica compile caches); `touch SERIAL-SERVE` = old serial start.
  ps1 also: PREP-CACHE on, gate delta_timeout 900 -> 240 s. Revert: `MODE=revert bash activate_ps1.sh`.
- ps2: grader uv env site-packages served from a byte-identical overlay copy via a symlink at the same path (gocryptfs was the grading
  bottleneck). Real-rollout row-by-row identity 300/300. Revert: `MODE=revert bash activate_ps2.sh`.
- `ops/probe_pod/probe_worker.py` (77081239, ps3): verifier probe_worker fbb15543 + setup-once (grader runtime readiness check once per
  worker process, GRADE-SETUP-ONCE.json) + live-math budget skip (capped reply -> 'neutral' without the native grader spawn, same outcome).
  Identity 300/300 vs fbb15543. Revert: `MODE=revert bash activate_ps3.sh` (restores fbb15543, closes the round).
- ps4: vLLM venv site-packages -> byte-identical overlay copy via symlink (`ps4_stage.sh`, `activate_ps4.sh`).
- `watchdog_ps2.py`: auto-revert on gate re-serve / serve_s > 320 / salvage / partial retry / epoch error / > 5 indeterminate per block.
- Tests: `test_skipred.py` (offline: closed round + head mismatch -> RED, delta wait), `ps_test.py` + `ps_compare.py` (saved-rollout grade identity).
