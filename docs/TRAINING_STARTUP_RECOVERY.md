# Explicit terminal startup-failure recovery

This private operator contract permits one replacement training request after a
proved fresh-source bootstrap failure. It does not retry an original job, amend
mining, discard accepted data, reset the optimizer or claim timely completion.
Original epoch timestamps, ten verifier admissions, failed job/envelope, PID/ticks
and empty original output namespace remain immutable. A recovered E13 remains
late; hourly measurements begin again with subsequent genuine epochs.

Configure `remote.training_startup_recovery_files` as an epoch-to-private-signed-
JSON map. Its ROOT-signed `terminal-training-startup-recovery-v1` payload contains
exactly: version, epoch, original_signed_job, original_job_sha256,
original_terminal, startup_witness, replacement_source_bundle,
replacement_job_label, created_at and expires_at. The original terminal binds
failed/nonzero exit, job ID, runner/child PID and ticks, started_at and finished_at.
The witness binds its observation time and full evidence SHA, names a
fresh-source-bootstrap-admission failure, declares no execution/CUDA/model load,
and confirms original processes absent, physical GPU idle and original output
namespace empty. These are independently observed operator attestations, not
miner execution proofs. Missing/ambiguous evidence holds recovery.

The controller preserves the original `<epoch>-train.json` record and reserves
one immutable recovery declaration. The replacement label is distinct and owns
its own original immutable job record/output namespace. Existing replacement
requests are adopted with their same signed bytes/capabilities; another
replacement declaration or terminal retry is forbidden. A fresh request must be
within the new explicit authorization lifetime (maximum 24 hours). Original
epoch start/deadline are unchanged. Once the replacement exists, its original
report-adoption rules apply even after the declaration expires.

The trainer consumes the exact originally authenticated compact objects and
receipts, refreshing GET URLs only before the new request. No verifier runs,
new receipts, success/failure relabeling or changes to batch population occur.
The full original computation manifest, runtime, steps, sampling, coverage,
hyperparameters and optimizer parent remain equal. Only execution source and
checkpoint GET capabilities may change. The original source qualification in
`trainer_state_binding` is validated against the authenticated original manifest;
the replacement source is independently signed and source-pinned. Mathematical
model/sampler/proof/optimizer modules must retain their original byte hashes.
The protocol admission module gains this explicit source transition but does not
change optimizer mathematics or descriptor layout. Latest durable optimizer
parent must still match before any fresh replacement dispatch.

This path is opt-in and rejects v1 execution-amendment mixing or recursive
recovery. The signed recovery module is mandatory in worker source inventory and
is reloaded through verified fresh source after pure admission. Normal compact
and persistent training remain unchanged. Scientific report, independent state
readback/publication and monotonic parent commit checks remain mandatory.
