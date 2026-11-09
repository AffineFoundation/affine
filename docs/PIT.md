# SN120 per-item train+eval (PIT) — serverless

PIT rents one GPU box **on demand** per queued registration. That box trains the
miner's batch with trainer-v834e from the bound head + fp32 master, then runs
the live-gate probe on the same box. The host scores the probe and records
incentives. Only the validator host stays always-on; baseline, bench and prior
are also short rent-per-job boxes (SPEC).

## Hard rule: on-demand rent only

- No warm pool, no pre-rented or idle PIT boxes.
- Empty queue ⇒ **zero** item rentals.
- B queued items with `max_boxes` N ⇒ at most **min(B, N)** concurrent item rentals.
- A box is released as soon as the item's result is stored or the item fails.
- Baseline / bench / prior jobs do **not** count toward the item cap.
- Controller default, not a config option (`warm_pool` / `pre_rent` refused).

## Live defaults (SPEC)

| Knob | Value |
|------|-------|
| `mode` | `shadow` (observe) or `live` (cutover) |
| Item `max_boxes` / `box_cap` | **5** live, **2** shadow |
| GPU | **B300 pinned** for item + baseline (no B200/H200 fallback) |
| Price ceiling | **$56/h** |
| Master upload | **after GREEN only** |
| Lium auto-backup | **disabled** on every rental |
| Trainer fallback | kept **24 h** after live cutover, then release with Jacob OK |
| REDs | **earn nothing** (last-10-GREEN rule) |

## Rules (short)

1. On dequeue, bind the item to the **current head** (and its expected fp32 master digest).
2. If `M(H)` is unpublished, dequeue **pauses**.
3. The **first fresh GREEN** to reach a decision becomes the new head (uploads bf16 `C` + `M(C)`, host applies via `pe_apply` / HEAD-CHAIN).
4. Later GREENs bound to an older head are **paid** but **`merge_blocked`** (Option B). Non-GREEN boxes destroy their state.
5. Shadow mode never writes `DECISION.json`, weights, `HEAD-CHAIN`, or R2 head pointers.
6. Never promote on a timeout, partial probe, or fail-closed error.
7. Box lost after SEALED ⇒ **full redo** with the same binding (SPEC D3).

## Gate math (unchanged)

k_cand 8, head rollouts 12, strict z>1, max output 65536, bf16, no fp8, N=1000.
fp32 master checks stay fail-closed. PIT never passes `--fp32-bootstrap-override`.

## Sibling jobs

| Job | Trigger | Cap | Notes |
|-----|---------|-----|-------|
| Baseline | new bindable head / round rollover | 2 | Secret draw stays host-private until decision; B300 |
| Bench | each GREEN_FRESH | 1 | Profile `qwen3.5-family-pp0`; merge-only history; coalesce to newest on backlog |
| Prior | each GREEN_FRESH | 1 | Cheap per-head pass; does not block scoring |

## Modules

| File | Role |
|------|------|
| `binding.py` | Dequeue + bind; pause on unpublished master |
| `expected_master.py` | Chain override → manifest → index → TRAINING-RESULT |
| `box.py` / `train_box.py` | Box phases + on-box train asserts |
| `controller.py` | On-demand rent, live/shadow decide+apply |
| `apply_live.py` | GREEN_FRESH → verify uploads → pe_apply; destroy otherwise |
| `cutover.py` | Live cutover / revert plan (trainer, warm, lanes) |
| `baseline_job.py` / `bench_job.py` / `prior_job.py` | Sibling rent-per-job services |
| `decisions.py` | Hash-chained decision log |
| `signer.py` | Ephemeral key + `HostRootSigner` / `FixtureRootSigner` |
| `pe_rent.py` / `pe_lane.py` / `pe_score.py` | Git ports; host RA modules when present |
| `ops/jobs/` | Shared job runner (price ceiling, auto-backup off, caps) |

`pe_score.score_epoch_shadow` always records the **box** candidate hash, never
the live epoch's `candidate`.

## Operator: shadow mode

```json
"parallel_eval": {
  "pit": {
    "enabled": true,
    "mode": "shadow",
    "max_boxes": 2
  }
}
```

Confirm: decision log under staging; **no** `DECISION.json` / `HEAD-CHAIN` / R2
head writes; rentals drop to 0 when the queue is empty.

## Operator: live cutover

1. Shadow pass criteria met (DESIGN §10 / SPEC §7 step 0).
2. ROOT admission: configure `AFFINE_PIT_ROOT_SEED_FILE` (mode 0600) or inject
   the host ROOT Identity into `HostRootSigner`. Never a bootstrap override.
3. Flip:

```json
"parallel_eval": {
  "pit": {
    "enabled": true,
    "mode": "live",
    "cutover": true,
    "max_boxes": 5,
    "master_upload": "after_green"
  }
}
```

4. Effect: trainer dequeue **off** (pod kept 24 h as hot fallback); warm pool
   **off**; standing PE lanes **off**; PIT owns the queue; first fresh GREEN
   uploads and applies; REDs earn nothing.
5. After 24 h of healthy live: release the trainer pod with Jacob OK.

## Operator: revert

```json
"parallel_eval": {
  "pit": {
    "enabled": false,
    "mode": "live",
    "revert": true
  }
}
```

Restores trainer dequeue, warm pool and standing lanes. Re-rent standing pods
with `replace_pod.py` if they were released (≤ $35/h or prior price + 10%).

## Tests

```bash
PYTHONPATH=. python3 -m unittest \
  ops.parallel_eval.pit.test_state_machine \
  ops.parallel_eval.pit.test_binding \
  ops.parallel_eval.pit.test_decisions \
  ops.parallel_eval.pit.test_signer \
  ops.parallel_eval.pit.test_expected_master \
  ops.parallel_eval.pit.test_ondemand_rent \
  ops.parallel_eval.pit.test_pe_score \
  ops.parallel_eval.pit.test_shadow_controller \
  ops.parallel_eval.pit.test_box_fake \
  ops.parallel_eval.pit.test_live_controller \
  ops.parallel_eval.pit.test_cutover \
  ops.parallel_eval.pit.test_jobs
```
