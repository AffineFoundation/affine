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
retirement and final/current checkpoint protection. This is a retention primitive;
an automatic trainer archive-and-retention supervisor is not yet deployed.
