# SN120 per-item train+eval (PIT)

PIT rents one GPU box per queued registration. That box trains the miner's batch
with trainer-v834e from the bound head + fp32 master, then runs the live-gate
probe on the same box. The host scores the probe and records incentives.

## Rules (short)

1. On dequeue, bind the item to the **current head** (and its expected fp32 master digest).
2. The **first GREEN** for that head becomes the new head.
3. Items dequeued **after** that GREEN bind to the new head.
4. Later GREENs still bound to an older head are **paid** but **`merge_blocked`** — never applied.
5. Shadow mode never writes `DECISION.json`, weights, `HEAD-CHAIN`, or R2 head pointers.

## Defaults (DESIGN §12)

- Baseline option A: one secret draw per head epoch via the existing baseline pod.
- Per-box ephemeral ed25519 key; ROOT-signed short-lived admission cert.
  This repo ships the verifier and the admission-request builder only. It never
  generates, embeds, or requests a real ROOT signature.
- M(C) (candidate fp32 master) uploaded during the probe.
- Box cap configurable; default **2**.
- Trainer pod kept as fallback.

## Gate math (unchanged)

k_cand 8, head rollouts 12, strict z>1, max output 65536, bf16, no fp8, N=1000,
same estimator. fp32 master checks stay fail-closed (`Fp32MasterError`, expected
master hash). PIT never passes `--fp32-bootstrap-override`.

## Enable shadow mode (operator)

1. Confirm this PR is deployed to the validator checkout you intend to run
   (shadow only; no live cutover).
2. In the PE / promotion-gate flag JSON, set:

```json
"parallel_eval": {
  "pit": {
    "enabled": true,
    "mode": "shadow",
    "box_cap": 2
  }
}
```

3. Leave every other gate setting alone.
4. Point host adapters at the machine-local `pe_rent` / `pe_lane` / `pe_score` /
   `pe_live` modules if you want real rental (optional for a dry shadow of the
   state machine; unit tests use stubs).
5. Run the host controller tick loop against staging state. Confirm:
   - decision log appends under the PIT staging root
   - **no** new `DECISION.json` / `HEAD-CHAIN.json` / R2 head pointer writes
6. Keep `enabled=false` (or unset) to disable.

## What this PR does / does not do

**Done:** state machine, binding/staleness, hash-chained decision log, box runner
ports, host controller, ephemeral admission request+verify, shadow guard, unit
tests, docs, flag defaults.

**Stubbed / host-only:** real `pe_rent`/`pe_lane`/`pe_score`/`pe_live` calls
(adapters only), real ROOT admission signing, live apply / R2 publish, wiring
into the production systemd units.

**Not in this PR:** live mode, gate-math changes, trainer fp32 weakening.
