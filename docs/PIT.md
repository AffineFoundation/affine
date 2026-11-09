# SN120 per-item train+eval (PIT)

PIT rents one GPU box **on demand** per queued registration. That box trains the
miner's batch with trainer-v834e from the bound head + fp32 master, then runs
the live-gate probe on the same box. The host scores the probe and records
incentives.

## Hard rule: on-demand rent only

- No warm pool, no pre-rented or idle PIT boxes.
- Empty queue ⇒ **zero** rentals.
- B queued items with `max_boxes` N ⇒ at most **min(B, N)** concurrent rentals.
- A box is released as soon as the item's result is stored or the item fails.
- This is controller default behaviour, not a config option (`warm_pool` /
  `pre_rent` flags are refused).

## Rules (short)

1. On dequeue, bind the item to the **current head** (and its expected fp32 master digest).
2. If `M(H)` is unpublished, dequeue **pauses**.
3. The **first fresh GREEN** to reach a decision becomes the new head.
4. Later GREENs bound to an older head are **paid** but **`merge_blocked`**.
5. Shadow mode never writes `DECISION.json`, weights, `HEAD-CHAIN`, or R2 head pointers.

## Defaults (DESIGN §12)

- Baseline option A: one secret draw per head epoch via the existing baseline pod.
- Per-box ephemeral ed25519 key; ROOT-signed short-lived admission cert.
  Repo ships envelope + verifier + **FixtureRootSigner** test double. Real ROOT
  signing stays operator-out-of-band (`request_real_root_signature` refuses).
- M(C) uploaded during the probe by default.
- `max_boxes` configurable; default **2**.
- Trainer pod kept as fallback.

## Gate math (unchanged)

k_cand 8, head rollouts 12, strict z>1, max output 65536, bf16, no fp8, N=1000.
fp32 master checks stay fail-closed. PIT never passes `--fp32-bootstrap-override`.

## Modules

| File | Role |
|------|------|
| `binding.py` | Dequeue + bind; pause on unpublished master; rebind before SEALED |
| `expected_master.py` | Chain override → manifest → index → TRAINING-RESULT |
| `box.py` / `train_box.py` | Box phases + on-box train asserts |
| `controller.py` | On-demand rent, watchdog edges, shadow scoring |
| `decisions.py` | Hash-chained decision log |
| `signer.py` | Ephemeral key + admission envelope/verify + fixture ROOT |
| `pe_rent.py` / `pe_lane.py` / `pe_score.py` | Git ports; load host RA modules when present |
| `shadow.py` / `config.py` / `state.py` | Guard, flag, DESIGN state machine |

`pe_score.score_epoch_shadow` always records the **box** candidate hash, never
the live epoch's `candidate`.

## Enable shadow mode (operator)

1. Land this PR on the validator checkout (shadow only).
2. In the PE / promotion-gate flag JSON:

```json
"parallel_eval": {
  "pit": {
    "enabled": true,
    "mode": "shadow",
    "max_boxes": 2,
    "master_upload": "during_probe"
  }
}
```

3. Leave gate math alone.
4. Optional: set `pe_rent_module` / `pe_lane_module` / `pe_score_module` to host
   RA module paths. Without them, unit/shadow uses synthetic rent + injected score.
5. Confirm: decision log under staging; **no** `DECISION.json` / `HEAD-CHAIN` /
   R2 head writes; rentals drop to 0 when the queue is empty.
6. Leave `enabled=false` (default) to keep PIT inert.

## Tests

```bash
PYTHONPATH=. python -m unittest \
  ops.parallel_eval.pit.test_state_machine \
  ops.parallel_eval.pit.test_binding \
  ops.parallel_eval.pit.test_decisions \
  ops.parallel_eval.pit.test_signer \
  ops.parallel_eval.pit.test_expected_master \
  ops.parallel_eval.pit.test_ondemand_rent \
  ops.parallel_eval.pit.test_pe_score \
  ops.parallel_eval.pit.test_shadow_controller \
  ops.parallel_eval.pit.test_box_fake
```
