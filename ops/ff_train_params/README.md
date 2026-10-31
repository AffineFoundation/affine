# Free-file per-turn lr / kl (affine-batch-v2), trainer v8.3.3j

Jacob 2026-10-31 07:08 CR (pt902u). Live from epoch 362. Built on trainer v8.3.3i (ops/renorm_step).

- Format: `affine-batch-v2` = v1 + optional top-level `"train": {"lr", "kl"}` (both optional; omitted = default).
  lr in [1.25e-7, 1e-6] (default 5e-7), kl in [0.005, 0.1] (default 0.02), inclusive. Out of range / non-number / extra key
  = invalid file (skipped). Never clamped. v1 unchanged; `train` in a v1 file is invalid. `release/ff_format.py`.
- Lifecycle (`release/ff_lifecycle.py`): v2 accepted only while `state/FF-TRAIN.json` is enabled; the effective params go into the
  signed sn120-ff-dispatch-v1 payload `train` {lr, kl, given, defaults} and K1-TURN (private). Flag off = exact previous behaviour.
- Trainer v8.3.3j (`patch_v833j.py` on the signed v8.3.3i host; `host-v833j/`): m = lr / 5e-7;
  n0_t = r_t ||d_t|| (the v8.3.3i per-tensor norm), c_t = 2e-5 (1 - 1e-6) ||M_H,t||, n_t = min(m n0_t, c_t), s = min(1, 1.0 / ||n||),
  Delta_t = s n_t g_t / ||g_t||. m = 1 runs the v8.3.3i code path unchanged. kl replaces kl_coefficient for that one update
  (L = loss_scale (1/N) sum_i (1/K_i) sum_t [pg + kl k3]); the reported config stays the signed one. Report: training.train_params.
- Public: turn.json training.train_params / turn.train / dispatch.train and mining.json training.miner_train_params.recent_green_turns,
  GREEN only (`services/`, `side/patch_sidecars.py`); batch_file.train in mining.json (`side/make_ff_public.py`); docs (`side/web_docs.py`).
- Activation: `activation/activate_tp.sh` (stage | switch | revert), auto-revert `activation/watchdog_tp.py`.
  Revert: `MODE=revert bash RA/20261031-ff-train-params/activation/activate_tp.sh`.
- Tests: `tests/` (step math vs an independent reference, m = 1 bit-identical to v8.3.3i, format v1 regression + v2, e2e selection
  simulation with a v1 file, a valid v2 file and an out-of-range v2 file), `activation/test_watchdog_tp.py`, `side/test_k1t.py`.
