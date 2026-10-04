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

The operator host also provides `ops.automatic_submission_retention`, a bounded
oneshot suitable for a systemd timer. It follows the authenticated current writer
pointer, supports original and newly approved controller workspaces, and keeps
a durable per-job/report ledger. Each run checks actual replica presence before
streaming R2 data. Its two-worker maximum and per-worker quota bound each pass;
worker infrastructure errors are recorded without miner penalties. This is
completed-download retention, not model-cache or bucket-object deletion.

```
PYTHONPATH=. .venv/bin/python -B -m ops.automatic_submission_retention \
  --original-config PRIVATE_ORIGINAL_CONTROLLER_CONFIG \
  --future-config PRIVATE_NEW_CONTROLLER_CONFIG \
  --writer-pointer PRIVATE_AUTHENTICATED_WRITER_POINTER \
  --authority OPERATOR_PUBLIC_KEY --output PRIVATE_RETENTION_RECEIPTS \
  --per-worker 8 --apply
```

The deployed operator timer runs again two minutes after the previous pass
finishes. It leaves the original signed requests, local reports and all R2
objects intact. It never changes approved scientific source on a worker.

Operators can run a single bounded pass or a continuous watcher:

```
PYTHONPATH=. .venv/bin/python -B -m ops.retain_verifier_downloads \
  --config PRIVATE_CONTROLLER_CONFIG \
  --writer-cutover SIGNED_WRITER_CUTOVER --authority OPERATOR_PUBLIC_KEY \
  --output PRIVATE_RETENTION_RECEIPTS --per-worker 8 --watch --interval 300
```

The watcher holds an exclusive local lock, reloads and authenticates the approved
worker/source roster each cycle, and considers only completed verifier jobs,
including earlier epochs. It inspects actual replica presence before downloading
archives and processes at most eight files per worker per cycle, with at most
four concurrent workers. Without `--watch`, a failed worker operation produces
a nonzero exit. A watch cycle retains failure evidence and retries through fresh
presence/archive checks on the next interval; it does not infer fraud or replay
scientific jobs. Empty queues and epoch boundaries are safe no-ops. Six additional
controls cover complete streamed archive checks, corruption and truncation,
boundary no-ops, signed roster disagreement and private configuration admission.
