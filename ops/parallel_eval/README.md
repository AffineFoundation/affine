# Generation-aware PE staging (v4)

Still NOT live, NOT activation-ready, and NOT a real GRPO/GPU replay.

New in v4:
- Master entries are `<fp32-master-digest>-<generation-id>`; signed head identity
  binds checkpoint ID + fp32 master digest + generation ID. Equal bf16 weights
  and even equal masters in distinct generations no longer alias.
- `install_generation_hooks` is a staged native trainer restore/GC hook. It
  selects an authenticated generation and reloads its residuals rather than
  trusting the native bf16-only memory cache. Missing/corrupt selections fail
  closed, never bootstrap-upcast. It is NOT installed on any live trainer.
- `R2Objects` uses an existing host-provided native Bucket; writes only an
  isolated private replay prefix. Both model-only shards and exact fp32 master
  metadata/shards are uploaded, fully GET-read and SHA/length checked. No
  credentials or presigned URLs are transported by this package.
- Signed restore validates generation identity, original signed merge request,
  projection and fresh-moments marker; verification needs public keys only.
- 22 CPU tests pass, including actual native trainer digest/GC functions,
  equal-bf16/different-master and equal-master/different-generation fixtures,
  memory-cache invalidation, same-checkpoint GREEN head advance, R2 corruption,
  and fresh/stale/CURRENT/K1 simulation cases.
- Separate on-host evidence proves actual private R2 publication/readback and a
  new-process public-key-only restore of a small nonpayable CPU fixture. This
  is NOT a fresh historical GRPO or GPU probe, and simulated K1 is not a real
  ledger/served join.

Still required: actual legacy lifecycle/controller effect adapters, real gate
ownership/draw/scorer composition, real K1/ledger/quiescence joins, admission of
these hooks/generation type in a staged trainer release, isolated rented-node
GRPO/GPU replay, and real drained activation qualification. Activation remains
an explicitly nonqualifying simulator. Nothing below overrides that boundary.

## Previous v3 implementation notes (historical)

# Parallel eval transaction staging — NOT LIVE / NOT activation-ready

This package is an isolated staging implementation. Every writer requires a
marked, nonempty-safe `Stage` root. There is no production root default, SSH,
GPU rental, chain action, service command, live lifecycle install, or public
policy execution. The publication adapter accepts **private, nonpayable**
policies only. `activation.py` operates on a fake service image, not systemd.

## Implemented

- `protocol.py`: durable claim/pending/keep/pin/restore/dispatch/apply outboxes,
  expiring worker leases, authenticated ACKs and replay-safe state. Slow receipt
  verification is outside the state lock. Workers and eval watchdogs must be
  independently scheduled; reconciliation itself does no network work.
- `inbox.py`: authenticated, idempotent lifecycle-event consumer. Source records
  stay immutable; a bad event does not prevent other lanes from progressing.
- Immutable per-head draw/baseline binding, no eight-turn refresh, scoring against
  old closed rounds, GREEN-time ordering, staleness up to two, durable CURRENT
  acknowledgement before head-chain/round advancement, and baseline creation for
  each successor head. The dispatch boundary is durable dispatch enqueue.
- RED/K1 served records; probe guards, exhausted infrastructure, restore timeouts,
  missing masters and stale overflow yield no-decision/retry grants, bound to the
  original K1 turn digest. Abandon requires authenticated **quiescence**, not just
  a revoked lease; a late remote write must be fenced before granting another turn.
- `masters.py`: native `_digest_map`-compatible bf16/master digests, safetensor
  inventories, projection validation, immutable store install and non-expiring
  owner pins. The staged GC hook shares the writers' lock and protects even
  malformed durable markers. Current-head pins and generation pins retire only
  after a successor is durable/pinned; pending eval pins survive independently.
- `publication.py`: signed merge requests and publisher receipts using the real
  Ed25519 verifier, real `ReleasePolicy.authorize_checkpoint`, complete local
  stored-model SHA readback, fp32 deltas, explicit merge generations with optimizer
  markers, immutable restoration and integrity verification.
- `activation.py`: drained/rechecked activation simulation, surgical PE-key and
  drop-in changes, verification, rollback and crash recovery. Concurrent epoch,
  CHUNK configuration, 64K caps and unrelated drop-ins are never restored wholesale.
- `archive.py`: read-only verification of real historical worker/publisher
  signatures and cached model files. It does **not** claim a fresh train/eval replay.

## Tests and qualification boundary

From a checkout, with the normal project Python dependencies:

```
python -B -m unittest ops.parallel_eval.test_protocol -v
```

Set `PE_NATIVE_TRAINER_SOURCE` to an ordinary copy of the trainer source to run
native compatibility tests. The test extracts only `_digest_map`, `_sha_tensor`,
`_fp32_load`, `_fp32_gc`, and `_fp32_store_entries` into a fresh test module; it
never imports/patches the running trainer. Without this input that test is skipped
and must **not** count as native qualification.

Signatures are real, but identities, tensors, gradients, draws, baselines, eval
workers, and service images in these tests are CPU/nonpayable fixtures. There is
no claim that a toy gradient is host GRPO, that a lease test is a new GPU pod-loss
experiment, or that a local filesystem object store is R2. No key material is
persisted. A green test suite cannot enable live operation.

## Exact remaining production work

1. Connect the existing admitted FF/v829t/v829v lifecycle and gate to these
   consumers, with claim-before-early-gate, original registration ordering,
   actual candidate keep and fenced restore/status receipts. The legacy
   `pe_lifecycle.py` does NOT yet emit this package's signed event format.
2. Admit the PE merge generation/restorer and durable GC hook in a trainer release
   without changing the training math or the separately deployed chunk settings.
   The PE transaction is intentionally **not** forged into a native gradient
   `Transaction`; the old restorer does not admit its new generation type.
   Replace private/local publishing with authenticated R2 publication/full readback.
3. Resolve native bf16-key aliasing: distinct fp32 masters can round to the same
   bf16 tensor digest. This store rejects that collision instead of losing a
   residual. Production needs generation/master-digest-aware selection and cache
   invalidation; checkpoint hash alone is not enough. Repeated checkpoint heads
   likewise fail closed pending a generation-aware head adapter.
4. Wire actual round creation/baseline, exact live probe/scorer, pinned H100+ runtime,
   final gate ledger and original K1 grant/served joins. Implement adapter-level
   effect idempotence and fencing; a command ACK is not itself an action.
5. Run the full signed **past-epoch real GRPO → eval pod → score → fresh/stale
   publish/restore → CURRENT → K1/ledger** replay, including pod loss, partial
   results, old-round scoring and both CURRENT crash gaps. CPU and archive tests
   below are necessary evidence, not that replay.
6. Qualify a real drained lifecycle activation adapter and surgical revert under
   admission. The current simulation always reports `ready_for_live=false`.
   A production CURRENT CAS/lock and command fencing are mandatory; the supplied
   private filesystem verifier is not a substitute.

No existing live readiness interlock should consume CPU fixtures as production
evidence. Do not change live flags or deploy this package from its README.
