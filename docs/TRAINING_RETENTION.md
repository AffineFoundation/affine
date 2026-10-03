# Completed trainer replica retention

`ops.training_retention.remove_training_replica` accepts a narrow operator plan
for a completed job's downloaded submission ZIP or archived checkpoint export.
The operator must authenticate the original job and report, verify every archived
object by reading its full bytes, and bind the exact local evidence files before
issuing the plan. Intermediate exports need their own authenticated archive;
their presence in a completed job directory does not establish an archive.

The helper independently checks the original operator signatures, job/report
binding and actual successful runner completion. Both original process identities
must have exited. A completed report alone cannot authorize deletion while the
trainer is still running. Downloads are bound to their original submission
position and hash. Exports are bound to their original job and step, exact complete
file map, content-addressed checkpoint ID and explicit protected checkpoints.
The final export must also match the original training report.

The GPU must be idle, local objects must have matching full hashes and sizes,
and files with hard links, open descriptors or memory mappings are refused.
Before retirement the helper rechecks original evidence and file identities.
Jobs, reports, runner records, archive objects and protected current checkpoints
remain intact. Infrastructure failures leave evidence for inspection; callers
must inspect actual completion before trying another operation.

Eight controls cover complete versus still-running jobs, original signatures and
byte bindings, substituted paths/hashes/positions, archive requirements, live
process identities, open and hardlinked files, GPU occupancy, intermediate export
retirement and final/current checkpoint protection.

`ops.retain_completed_training` provides the bounded operator supervisor:

```sh
.venv/bin/python -B -m ops.retain_completed_training \
  --config /private/current-controller.json \
  --writer-cutover /private/signed-source-approvals.json \
  --authority CONTROLLER_PUBLIC_HEX \
  --controller-process /private/original-controller-process.json \
  --output /private/trainer-retention --watch --interval 300
```

Each cycle authenticates the admitted original source, job, manifest and report,
checks the actual controller PID/start ticks and immutable configuration bytes,
and observes the trainer's original successful runner completion. It defers
while the trainer GPU is occupied. One cycle handles at most one completed job,
two intermediate exports and its bounded original submission population.
Intermediate exports are uploaded with scoped per-object capabilities, then
independently read back in full before authenticated archive descriptors and
retirement plans are issued. Conditional writes preserve existing archives.
The remote helper checks actual successful process completion again before
retirement. Current and pending successor checkpoints are protected.

The supervisor holds an exclusive operator lock and records private progress
and actual completion. A transport, integrity or process failure stops it for
inspection; it does not restart a GPU job or infer completion from an observation
timeout. Model workers receive no permanent bucket or operator signing credential.
Final learned exports and model caches are retained, so this removes temporary
training replicas rather than bounding the entire model history. Final-checkpoint
and cache retention remain separate work.

Six supervisor controls include real isolated Python/HTTP archive and retirement
execution, a corrupted archive that prevents all deletion, original source and
receipt admission, stream truncation, and live controller/configuration guards.
