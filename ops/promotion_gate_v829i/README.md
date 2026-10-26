# SN120 checkpoint promotion gate (lifecycle host v8.2.9i) + public training params

Approved by Jacob 2026-10-26 (pt682u design, t684u go-live, t685u training params). Live from epoch 152
(3:04 PM CR, UTC-6).

* `mimo_v829_train_gate_launcher.v829i.py` / `v829i.diff`: v829h + `install_v829i_promotion_gate`. Behind
  `staged/PROMOTION-GATE.json` (`enabled: true`). GREEN -> promote; RED -> no-update onto the head
  (`reason promotion-gate-red-...`); no decision after `decision_timeout_seconds` -> fail-open promote.
* `gate_core.py`: private sample draw, commitment, paired z / cap-hit decision (z > -1, cap < 0.20, >= 200 tasks).
* `gate_service.py` (`affine-promotion-gate.service`): per epoch draws the private sample, keeps the head generation
  on the trainer pod, probes the candidate on sample minus batch on the probe pod, writes DECISION.json, restores the
  trainer on RED, publishes commitments/verdicts to https://affine.io/gate/index.json (seed revealed 2 epochs later).
* `pod/`: `gate_fetch.py` (probe pod checkpoint fetch), `trainer_keep.py` (trainer pod keep/restore).
* `probe_daemon.gate.diff`: env-probe daemon yields the probe pod while the gate is busy.
* `flip/`: flip scorer explicit RED = zero credit guard + lineage test (RED-then-GREEN, rollback composition).
* `training/`: `training_params.py` (`affine-training-params.service`) builds the additive mining.json `training`
  block and https://affine.io/training.json from the live signed lifecycle config, trainer host config, CURRENT and
  the epoch manifest; `observer.training.diff` merges it into mining.json when epoch + head match.
* `backtest/`: e125-e150 backtest (20 and 100 seeds).

Revert: set `enabled: false` in `staged/PROMOTION-GATE.json` (instant, old behaviour); full revert
`REVERT=1 FROM_EPOCH=<N> activate_v829i.sh` (back to release-v829h-collect1200-signed).
Excluded: seeds, private samples, signed private operator configs, keys.

## Verdict ledger and site (Jacob t690u)
* `gate_ledger.py` (`affine-gate-ledger.service`): append-only `state/LEDGER.public.jsonl` -> https://affine.io/gate.json
  (epoch, H, C, promoted/rejected, z, cap-hit, n, reason, gather/train/pre-probe/post-probe/decision timestamps) and a
  private `REJECTED.private.jsonl` (rejected candidate id, TRAINING-RESULT/DISPATCH/JOB sha256, descriptor, R2 prefix).
* `site/site-metrics.js` (+ `site-metrics.gate.diff`): "Post-probe (gate)" segment and pre/post probe times + verdict in the
  Epoch timings chart; new "Promotion gate" panel (chart and table views, expand button).
* `gate_service.py` no longer reveals the sample seed or indices; only the commitment is published.

## First gated epoch (e152)
RED: candidate 8b9d5816 vs head 189de412, paired z -1.129, cap-hit 11.8%, n 965 (35 batch tasks dropped). Lifecycle finalized
no-update onto 189de412 (`promotion-gate-red-e152-8b9d5816-kept-189de412`), trainer pod restored in 30 s, e153 mines on 189de412.
Fix shipped during e152: a vLLM replica died mid-run (inductor-cache race), so the gate now re-serves dead replicas and
reruns partial candidate probes (`gate_reserve`, `gate_run_partial_retry`).

## Threshold change (Jacob pt692u, 2026-10-26 5:26 PM CR): pass unless clearly worse
From e154: GREEN iff paired z >= -1.5 (`z_min: -1.5`, `z_inclusive: true` in staged/PROMOTION-GATE.json, hot) and cap-hit < 0.20,
>= 200 tasks. e152-e153 keep their z > -1 verdicts. Backtest (100 seeds, `ZMIN=-1.5 ZINC=1 backtest.py 100`): flat/healthy
full-probe steps per-seed false-reject 6.3% (was 12.2% at z > -1); the 7 known collapses (128, 129, 140, 142, 144, 146, 148)
are still rejected on 100/100 seeds.
`monitor/flip_freeze.gate-red-expected.diff`: the weight-freshness monitor treats a frozen flip digest as expected (warning,
not failure) only when no new env-probe result finished since the last digest change and the span contains a gate RED or
no-update epoch.
