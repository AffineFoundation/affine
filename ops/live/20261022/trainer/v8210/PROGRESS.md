# 20261022-async-optimizer-save: trainer optimizer-state persistence off the critical path (v8.2.10)
Ordered by Jacob 2026-10-22: "saving the optimizer state should not be important can we remove this requirement".
Standing order: "Push forward with any changes you need to make the learning stable and working." (times UTC; CR = UTC-6)

## Decision: persistence DISABLED (not async)
- Considered the async every-N-epochs snapshot. Rejected as the riskier option:
  * it must read masters/moments while the next epoch can step them -> needs a barrier/cancel in CPUAdamW.step;
  * an N-epoch-old snapshot has stale fp32 masters; using it after a crash needs a NEW mixed recovery path
    (old moments + current bf16 weights), which does not exist and is not tested;
  * on this pod the sustained write rate is the problem (see DISK.md): 107 GB every N epochs would still saturate
    the disk for over an hour and slow the next epoch's model write, which IS on the critical path.
- Chosen: engine.optimizer_persistence='none'. The model is staged/hashed/published exactly as before (same code
  path, same bytes, same checkpoint ids); the optimizer generation is one ~1 KB marker optimizer/NOT-PERSISTED.json
  (binds step, input/output checkpoint, inventory digest, objective). The live optimizer stays in trainer memory (unchanged).
- Recovery: a generation with optimizer/descriptor.json (all legacy ones, incl. e91's) restores exactly via the
  original function; a marker generation rebuilds fresh AdamW moments with the step kept (masters = upcast of the
  exact published bf16 weights): the approved 2026-10-14 / e88 method, now automatic, with all original journal /
  report / receipt / model-inventory checks kept.

## What else needs the optimizer file or its hash? (checked)
- Lifecycle / TRAINING-RESULT / FINALIZED / verifier / audit / seal / DISPATCH: no. The signed report carries the training
  report, publication receipt (model objects only, optimizer_state_uploaded=false), transaction_report_sha256; none of
  them hash optimizer files (e90 TRAINING-RESULT inspected). Report schema unchanged.
- Pod-local transaction journal (TRANSACTION.json artifacts, STAGED.json): requires >=1 `optimizer/` artifact ->
  satisfied by the marker (journal code unchanged). STAGED.json keys unchanged (optimizer_descriptor_sha256 = digest of marker).
- Cold recovery (recovery.restore_transaction_state): required optimizer/descriptor.json -> wrapped (new pinned site).
- Retention (retire_obsolete_optimizer_states): rmtree of old optimizer dirs; works with the marker (tested).
- Arbos ops scripts: only the manual rebuild_trainer_state_pod.py reads optimizer state; nothing scheduled does.
- R2 publication request still says optimizer_state_location='trainer-local', optimizer_state_uploaded=false (unchanged).

## Release
- stage/trainer-host-v8210 -> release-v8210-signed (sign_v8210.py, ROOT 3301134b, seed never printed/copied).
  Main config optimizer_persistence=none sha f6e14d0d...; pre-signed in-place rollback variant -sync ('synchronous') 5e16fe0f...
- Tests: test/test_v8210_cpu.py (real patched execute_transaction + journal + retention + recovery, tiny bf16 model,
  both modes, legacy generation restored exactly through the wrapper): PASS on Arbos and on the pod runtime (torch 2.14).
- activate_v8210.sh --preflight 19:25Z: pod files verified, owned-worker --check both variants OK (nothing restarted).
- Armed 19:27Z: transient unit mimo-v8210-arm (activate_v8210.sh --arm 91) waits for epochs/91/TRAINING-RESULT.json,
  then drain-restarts the trainer onto v8.2.10 before e92 dispatch (refuses if a TRAINING-JOB is in flight or e91 needs recovery).
