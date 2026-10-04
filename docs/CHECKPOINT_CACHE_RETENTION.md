# Bounded model-cache housekeeping

`python -B -m ops.retain_checkpoint_caches` operates independently of the live
controller. It retires at most one obsolete model cache per cycle. Use private
`--config`, `--scopes`, `--controller-process`, `--authority` and `--output`
arguments, with optional `--watch --interval 300`. The scopes record binds its
explicit canonical cache roots to the exact controller config and each role's
endpoint digest. It does not recursively scan workspaces or inspect secret files.

The original controller PID/start ticks and config hash must still match. Current
and pending weights, the original signed active manifest, unfinished signed
verification jobs, and dispatched GPU jobs without checked completion reports
are protected. Expiry or an observation failure does not establish completion.

Only an idle role's content-addressed `checkpoint/<hash>` or `checkpoints/<hash>`
directory can qualify. The operator authenticates the authority-specific public
checkpoint descriptor and freshly streams and hashes every R2 object before
issuing a narrow remote plan. A missing authenticated descriptor leaves the cache
intact. A new GPU job during archive checking defers deletion.

The remote helper checks exact membership, local bytes, regular files, link
counts, open file descriptors, memory mappings, and GPU quiescence again before
renaming and removing that one replica. It retains R2 objects, current/pending
models, job records, reports, optimizer snapshots and training exports. Existing
training-export retention has a separate policy; it is not extended by this tool.
Original operation plans, full archive readbacks and remote completion receipts
remain private and inspectable. An uncertain operation stops the watch rather
than automatically retrying or restarting a worker.

The tests exercise signed unfinished-job protection, changed scope/config and
endpoint refusal, real complete archive reads and corruption rejection, and real
isolated helper deletion with current-cache and archive preservation. The
isolated unprivileged test scans its own actual `/proc` descriptors and maps;
production root workers use the helper's full host process scan. Existing helper
controls separately cover open files, closed-descriptor memory mappings, changed
bytes, extra members, GPU occupancy, symlinks and hard links.
Additional supervisor controls require busy-GPU deferral without archive access
and refusal when a new job reference arrives during full archive readback.

This reclaims obsolete replicas without changing scientific verification or
scoring. It does not establish cross-hardware qualification or a learning gain.
