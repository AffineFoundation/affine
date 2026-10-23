# Flip incentive (LIVE) + authenticity couple

**Status: LIVE.** On-chain weights use `sn120-flip-incentive-live-v1` at
https://affine.io/scores.json (`live_weights: true`). Miner-facing contract text
lives in `dashboard/public/llms.txt` under “Flip-incentive LIVE scoring —
October 23, 2026”, also served at https://affine.io/llms.txt.

## Client requirements

**No miner client change is required** for flip pay or the authenticity couple.

- Keep the October 13 MiMo complete-group contract: nonces 0–7, at least one
  completed correct and one completed wrong, same upload / TOPLOC / proof rules.
- Stock task search may stay a random shuffle over signed `environments[].indices`.
- Optional local preference of frontier / currently-failing tasks is strategy
  only; use `--indices` with an authorized subset if desired.

## Authenticity couple (credit + train)

Jacob rule (2026-10-23):

1. Keep requiring TOPLOC + sampling checks.
2. If a batch/traj **fails** TOPLOC or sampling verification → **do not train
   (GRPO)** and **do not award flip credit**.
3. Unverified / unaudited / pending → same: **nothing** for credit and train.
4. Flip formula has **no** authenticity bonus (`no_authenticity_guard=true`).
5. Verify **all** batches that can earn flip credit or enter GRPO (not
   spot-check-only for eligibility).
6. Do **not** loosen TOPLOC tolerances.
7. **Never produce / never submit all-zero weights.** Authenticity and
   `unknown_policy` gates must retain last-good shares (or refuse submit and
   keep on-chain). Never burn. Never owner-only.

In-repo wiring:

| Piece | Role |
| --- | --- |
| `subnet/mimo_training_authenticity.py` | Filter sealed `selected[]` → train pack (`verified_valid` only) |
| `subnet/mimo_lifecycle_adapters.make_training_job` | Publishes `mimo/train-pack.json`; strips unaudited/fail |
| `subnet/mimo_grpo/prepared.py` | Trainer refuses packs without sampling authenticity |
| `subnet/flip_share_floor.py` | Last-good share floor; refuses all-zero submit |

Sealed coordinator `dispatch.json` remains the credit/audit join surface and may
still record native-graded unaudited assurance. GRPO consumes only the filtered
train pack.

## Live formula flags (ops)

Public metadata on https://affine.io/scores.json:

- `formula.version`: `sn120-flip-incentive-live-v1`
- `formula.no_authenticity_guard`: **true** (no bonus; keep true)
- `formula.sampling_verify_gate.enabled`: true
- `formula.sampling_verify_gate.fail_strips_credit`: true
- `formula.sampling_verify_gate.credit_requires_verify_pass`: true
- `formula.sampling_verify_gate.unknown_policy`: must keep a never-all-zero path
  (`pass_through` / `strip_unverified` / `zero_unverified` + share floor)

### ArbosLife ops flags to align after this PR

1. **Keep** `no_authenticity_guard=true` (do not add an authenticity weight term).
2. **Keep** `sampling_verify_gate.fail_strips_credit=true` and
   `credit_requires_verify_pass=true`.
3. **Enable** train-denylist / sampling-verify ledger feed into the MiMo
   lifecycle (`Services.sampling_verify_ledger`, `Services.train_denylist`) so
   GRPO packing strips fail + unverified (this PR wires the consumer).
4. **Require full verify** for any batch that can earn flip credit or enter the
   train pack (not spot-only eligibility).
5. **Wire share floor** (`subnet/flip_share_floor.apply_share_floor`) in the flip
   score → hourly writer path so authenticity/`unknown_policy` never submits
   all-zero weights; on refuse, retain on-chain.
6. **Never** enable burn or owner-only fallback writers.

## Optional read-only audit helper

```bash
.venv/bin/python ops/flip_incentive_audit.py summary
.venv/bin/python ops/flip_incentive_audit.py suggest-indices --limit 64
```

Fetches public scores and lagged `/probe/tasks/` dumps. Does not write chain
weights or alter the signed miner source.

## Related URLs

- https://affine.io/llms.txt
- https://affine.io/mining.json
- https://affine.io/scores.json (LIVE flip)
- https://affine.io/scores-shadow.json (optional mirror)
- https://affine.io/probe/index.json · https://affine.io/probe/series.json
