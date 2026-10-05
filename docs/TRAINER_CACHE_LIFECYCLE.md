# Automatic trainer cache retirement

The persistent training controller starts CPU housekeeping only after the original
job's complete inference checkpoint and FP32 optimizer-state publication/readback,
authenticated lineage validation, and authority latest-state commit. The same hook
runs during authenticated recovery of an already completed original job. It does
not rerun training, change optimizer precision, or repeat model hashing/downloads.

The authority signs `durable-original-trainer-cache-ACK-v1`, binding the original
job and report hashes, approved input/output checkpoint inventories, committed
optimizer descriptor/namespace/counter, and explicitly owned trainer input path.
The remote helper checks that original job and report and its successful terminal
status. A live original runner or child defers cleanup. An inherited checkpoint
lease protects input files even if a new runner dies before its child exits.

Owned obsolete checkpoint members and receipted downloads are retired using their
already authenticated inventory and inode receipts. The current new checkpoint,
in-flight inputs, externally mapped paths, source, keys, reports, diagnostics, and
optimizer-state files outside this explicit checkpoint catalog remain intact.
A monotonic optimizer-counter marker prevents a late old acknowledgement from
retiring a newer current checkpoint. Routed cache and ownership journals remove
only paths the helper actually reports deleted.

Cleanup runs on a separate thread with a bounded remote transport call. Failure,
resource contention, or unavailable helper records a deferred/not-configured
result and does not invalidate training or block advancement to the next epoch.
Completed original-job cleanup is idempotent. Deferred cleanup can be retried on
authenticated recovery; it is not a background discovery scan of arbitrary files.
The one-shot controller process waits for its bounded housekeeping thread at exit.

New pinned workers contain `subnet/cache_lifecycle.py` and
`subnet/trainer_cache_lifecycle.py`. An existing historical worker can instead use
an operator-reviewed `cache_lifecycle_overlay` with exactly `code` and `files`;
`files` must contain exactly those two module paths and their SHA256 hashes. The
CPU housekeeping invocation authenticates both module hashes and uses the
historical backend source for original-job authentication. An overlay never
changes or relaunches the original training process. Preparing this source does
not deploy it or authorize deletion on any production machine.
