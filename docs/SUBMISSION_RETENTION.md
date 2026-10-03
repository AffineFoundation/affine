# Completed verifier download retention

Verifier job directories can accumulate duplicate submission ZIPs even though
the immutable submissions already live in R2. `ops.submission_retention`
constructs narrowly scoped removal plans from completed queue rows. It verifies
the original operator-signed job and manifest, the winning worker's signed report
request and lease token, report hashes, approved source membership, completion
window and frozen receipt bindings. Leased or unfinished jobs cannot qualify.

Before applying a plan, the operator streams and hashes every byte of the
immutable R2 archive and checks its size. On the worker, removal rechecks the
original local report, the exact regular download file, its full SHA256 and size,
canonical paths, open process file descriptors and stable inode metadata.
It unlinks only that duplicate ZIP. Reports, jobs, model caches and remote
archives remain intact. Missing replicas are idempotent; an unreadable process
descriptor or changed file stops removal. This requires a worker account that
can inspect its process namespace.

Seven controls cover original signatures and lease bindings, unfinished jobs,
unapproved source, archive checks, local corruption, symlinks, changed reports,
and a real open file. Two actual operator operations removed eighteen completed
download replicas totaling 9,520,312,600 bytes from the two live verifiers after
fresh full-body archive checks. Both operations finished successfully with no
reported failures. The second operation first inspected which replicas remained,
so it did not repeatedly select already removed downloads.

This bounds one source of accumulation. It does not yet provide automatic model
checkpoint, trainer intermediate-export, or evaluator tensor retention. Those
need separate admission and archive checks before a continuous fleet can claim
bounded long-term disk usage. Removing download replicas does not change scoring,
penalties, original jobs, deadlines, or scientific verification.
