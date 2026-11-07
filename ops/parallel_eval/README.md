# PE staged runtime — not live activation admission

Every filesystem adapter requires an explicitly marked isolated `Stage`; it
rejects production lifecycle/trainer roots. Nothing here installs a live gate,
changes a trainer release, stops a service, or modifies chain parameters.

## Implemented interfaces

- `Producer` / `Inbox`: immutable signed lifecycle events and durable ACKs.
- `EffectWorker`: per-command physical locks, authenticated receipt journaling,
  restart replay without repeating completed effects, and fenced lease renewal.
  An interrupted remote operation must reconcile its command id before retry;
  a local lock or expired lease is **not** remote quiescence.
- `LifecycleConsumers`: claim ACK before job creation; keep/master pins before
  native epoch handoff. A retained candidate returns native `no-update` with
  `pe_state=pending`, so the next restored-head job can train while GPU eval is
  still running. This is not a miner RED or a head promotion.
- `K1Join`: frozen native-schema captures and queue reservations, once-only
  committed turns, and infrastructure retry grants without spending a turn.
  Native FIFO/registration/grant logic must read `queue_rows()`; unresolved
  reservations count once and no-decision releases only its own reservation.
- `GateOwnership`: a real file-lock owner/generation fence. Every legacy early
  candidate mutation **and late apply** must enter its guard. Merely looking for
  a PE claim is unsafe. Transfer requires authenticated real drain/current proof.
- `MasterStore` / admitted `trainer_generation`: fp32-digest plus generation-id
  selection/capture; same-bf16 generations cannot alias; missing/corrupt masters
  never bootstrap. Pending owners are non-expiring and survive old PIN expiry.
- `PrivatePublisher` / `R2Objects`: signed explicit PE-generation publication,
  full model/master GET verification and restoration. Signed requests persist
  before effects; incomplete master copies are retained as evidence, not used.

## Still required before production eligibility

1. Compose the consumers/fence into the actual legacy lifecycle, FF producer,
   early gate and promotion entrypoints, under an admitted release. Freeze native
   K1 source bytes before claiming; native pre-training writes must reserve, not
   prematurely commit, a no-decision turn.
2. Qualify independent trainer train/restore fencing and permanent role identity
   recovery. The experimental ROOT continuation certificates preserve original
   pre-GRPO claims; they are not a production credential-rotation design.
3. Bind real draw creation/baseline/scorer to a generation-aware owner, not a bf16
   checkpoint id, including speculative pre-open and restarts.
4. Qualify native GREEN promotion at the actual CURRENT write seam and a quiet
   training boundary. A late GREEN must not consume an unrelated miner epoch.
   Fresh/stale apply, signature/readback, round closure, retention and crash-gap
   reconciliation all need the same durable generation-aware CURRENT evidence.
   `promotion_result()` deliberately fails closed until this is supplied.
5. Join actual native FINALIZED/CURRENT, ledger/K1 and authenticated remote
   quiescence. A fake service image is not a stopped process/cgroup proof.
6. Qualify a drained **real** lifecycle activation transaction. `activation.py`
   is explicitly a simulator and always reports `ready_for_live=false`.

Actual rented-GPU replay evidence and operational notes live on the validator
under `state/root-audits/20261106-parallel-cutover/`. CPU fixtures are never
reported as a real GRPO/GPU replay or a production activation qualification.
