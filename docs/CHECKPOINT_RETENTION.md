# Archived checkpoint cache retirement

`ops.checkpoint_retention.remove_checkpoint_replica` handles content-addressed
checkpoint cache directories separately from submission downloads. The operator
must authenticate the original checkpoint descriptor and freshly stream and
hash every R2 object before delegating a removal plan. The canonical file-map
digest must match the checkpoint ID. Current and active checkpoint IDs are
explicitly protected.

The worker requires an idle GPU and an exact local inventory of regular,
single-link files with matching sizes and full SHA256 hashes. It checks both
open descriptors and process memory maps, rechecks file identity and GPU state,
then retires the directory and removes only those checked files. Unreadable
process state, symlinks, extra files, hard links, changed bytes or active model
references stop retirement. Only ordinary `checkpoint/<id>` or
`checkpoints/<id>` caches qualify; job exports and optimizer snapshots are
excluded. The primitive does not delete reports, jobs or archived weights.

An actual October 3 operation authenticated checkpoint `94ae9c68…` and freshly
hashed every complete R2 object. The miner and evaluator retired two exact old
copies totaling 30,485,452,468 bytes, with no failures. The current f9 checkpoint,
job records and an independent verifier copy were preserved. Seven controls
cover protected/active IDs, archive admission, exact files, corruption, symlinks,
hard links, GPU occupancy, an open file and a real memory mapping after its file
descriptor was closed.

This is a narrow operator primitive, not a complete automatic model-history
policy. Unpublished intermediate optimizer snapshots still require separate
archival and qualification. Long-term bounded disk usage remains unfinished.

`remove_verified_hardlink_alias` handles one explicitly known two-path alias.
Both paths must name the same content-addressed checkpoint, contain exactly the
authenticated file map and refer to the same regular inodes with exactly two
links. Unknown additional links, current or active checkpoint IDs, missing full
archive readback, open descriptors, memory maps and GPU activity all refuse the
operation. It removes only the alias and preserves the source. That operation
reports zero physical bytes freed; the remaining source can subsequently use
ordinary retirement after another protection check.

Fifteen checkpoint controls, including seven real hardlink cases, pass; the
combined checkpoint, trainer and submission retention suite passes 33 controls.
The epoch-eight verifier cleanup observer has checked the two original cache
paths and is checking every R2 archive byte. Retirement remains conditional on
the successor becoming current and the old checkpoint being unreferenced. It
does not change job records, proofs, leases, model computation or payout policy.
