# SN120 MiMo trainer host v8.3.2: optimizer drift cap + gentler restarts

v8.3.2 = signed v8.3.1 trainer host + optimizer drift control. Only `host/mimo_host_trainer_v83.py` (the launcher) changed; see `v832-launcher.diff`.
Approved by Jacob on 2026-10-26 at 13:06 CR.

## Behaviour (engine flags in the ROOT-signed host config)
| flag | prod | code default (absent) |
|---|---|---|
| `drift_cap_rel` | `2e-5` | `None` (off) |
| `restart_lr_warm_steps` | `3` | `0` (off) |
| `restart_lr_scale` | `0.5` | `1.0` (off) |

- **Drift cap.** After each AdamW step, the fp32 master update of every tensor is scaled so that ||ΔW_t||/||W_t|| ≤ `drift_cap_rel`. One global factor then keeps ||ΔW||/||W|| ≤ `drift_cap_rel`, and the bf16 actor is re-projected from the capped masters.
  - The bound is re-measured after the change, with a correction for fp32 rounding. The step fails closed if the bound is not met.
  - Adam moments are not modified.
- **Gentler restarts.** When the AdamW moments are fresh (no state, or all-zero moments after a cold recovery), the LR is multiplied by `restart_lr_scale` for the first `restart_lr_warm_steps` optimizer steps.
- **Log line** (one per step): `{"component":"trainer-host-v832","event":"drift-control", ...}` with:
  - pre/post-cap relative drift, tensors capped, global scale, max per-tensor drift pre/post
  - lr_factor and steps since fresh
  - The same record is also written as `training.optimizer.drift_control` in the training report.
- **Rank consistency.** Only rank 0 owns the CPU AdamW. Helper ranks receive the resulting bf16 weights by `dist.broadcast(src=0)`, so all ranks are identical.
- **Unchanged:** grad clip (1.0), loss, data, scoring, weight writer, TOPLOC, nonce.

## Files
- `host/`: the full v8.3.2 host file set (the private signed configs are not published).
- `release/sign_v832.py`: ROOT signing. It asserts that only the launcher changed vs v8.3.1, the method pins are identical, and the v8.3.1 engine keys are unchanged. It signs an ON config and an OFF config.
- `tests/`: the CPU sanity tests and their outputs: synthetic tensors with the real CPUAdamW, plus gloo world-8 rank agreement.

## Revert
Use either of these:
- Run `ROLLBACK_v832.sh`, which removes the v8.3.2 supervised drop-in so the v8.3.1 owned worker comes back.
- Activate the signed OFF config (`owned-worker-v832-off.json`). It runs the same code with the flags absent, which is bitwise the v8.3.1 step.
