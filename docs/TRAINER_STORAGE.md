# Training job disk admission

## Active trainer-local Adam lifecycle (October 8, 2026)

Only inference checkpoints and authenticated reports are published to R2.
The trainer retains FP32 master weights, Adam moments and counters locally;
there are no optimizer tensor upload or download capabilities in this mode.
The matching local parent is mandatory. Missing state stops training without
resetting Adam or fetching a stale historical state.

The local parent is read in place through authenticated, unchanged owned file
descriptors. It survives worker failure and stays until the successor model is
published, verified and acknowledged. The cache then promotes the candidate
and automatically removes the superseded parent. Current/candidate byte caps,
exclusive leases and disk/RAM admission remain enforced.

The high-memory deployment enables `optimizer_cache_volume:
trainer-local-memory-v1` on the trainer endpoint. The launcher authenticates and
prepares an owned local shared-memory volume automatically. Current and candidate
snapshots alternate between disk and shared memory: only one optimizer snapshot
occupies shared memory, leaving room for live CPU Adam within the container RAM
limit. Candidate exports use the chosen volume directly, avoiding extra copies.
Promotion retires only the authenticated superseded files and unmounts the empty
owned memory candidate. Setup refuses live jobs, pending candidates, changed
files or insufficient container memory. Per-job admission accounts for both the
model disk and the optimizer volume, including shared-memory usage.
Later process restarts reuse the same working volume. Rebooting or losing this
trainer loses Adam; only the inference model is durable in R2.

Epoch 59 provides a measured end-to-end control: 194 tasks / 776 pairs advanced
Adam 48 to 49, with zero optimizer tensor uploads. Local state save took 139.6
seconds, compared with 575.5 seconds for epoch 58 export/upload. This is a stage
comparison, not an assertion that all epoch costs fell by that amount. Model
publication/readback took 129.4 seconds. Training arithmetic, sampler and miner
contracts were unchanged.

The prospective controller computes download space from the frozen receipts of
every miner whose audited batches enter its training request. Receipt sizes must
be positive bounded integers and match the audited submission hash; at most 256
submission files may enter a job. It counts each retained download, even if two
request entries share the same hash. Unselected uploads do not consume this
job's download reserve.

The trainer keeps those compressed ZIPs throughout the job, but decodes one
submission at a time. Admission therefore reserves the total planned compressed
bytes (at least one complete artifact budget), one raw-artifact working budget,
a policy-specific model snapshot/export allowance, any missing input
checkpoint and the fixed safety reserve. Previously the compressed reserve
covered just one submission even when the job downloaded many files.

For the exact covered-pair v3 policy, future admission reserves two model-sized
outputs: the final checkpoint and one complete temporary export. That trainer
does not save per-update checkpoints. Other or unrecognized policies retain
the conservative allowance of one checkpoint per update plus the final export.
The input download is still counted if no trainer-local cache is registered;
every selected ZIP and the raw workspace remain in the reserve. This change
does not reduce the accepted training population to fit available space.
Controls exercise 1, 3 and 32 updates, insufficient disk, a missing input and
policy substitution. The change is prospective and has not replaced source
3bacecbf or the running epoch-eight controller.

Five new controls cover selecting/counting frozen files, substituted hashes,
invalid sizes and populations, aggregate disk refusal and integer bounds.
Existing routing, remote-backend and capacity controls pass. This changes future
disk admission, not training mathematics or artifact/numerical budgets. Active
signed jobs retain their original code and capacity receipts. This source needs
future role admission before activation.

Completed trainer exports use policy-specific paths: legacy jobs retain
`checkpoint-step-N`, while the covered-pair policy writes one
`checkpoint-covered-final`. Housekeeping authenticates the original job,
report and actual runner completion before selecting either form. Current or
referenced checkpoints, open files and GPU use prevent retirement. Every
retired file must first be read back completely from its authenticated R2
archive and match its original local size and hash; job and report evidence
remain intact.

Unsharded BF16 exports can exceed the old 5 GiB per-object housekeeping bound.
Model `.safetensors` objects now have a 32 GiB bound; other checkpoint files and
submission downloads retain their 5 GiB bound. This admits the existing 7B
model's approximately 15 GB single-file export without weakening signature,
full-byte verification or active-checkpoint protections. These operator-only
changes do not alter pinned training jobs or numerical settings.
The ordinary completed-job and obsolete-final housekeeping entry points use
these same bounds; obsolete-final archives reuse the cache-retention reader's
signed-descriptor and complete streaming verification.

The 32 GiB model bound governs authentication and retirement of an existing
archive. It does not increase R2's single-request upload limit. New legacy
unsharded exports larger than 5 GiB require multipart upload; the ordinary
intermediate uploader currently issues single-object PUT requests. Current
fixed/covered exports explicitly use 4 GB shards and fit that transport. See
[Cloudflare's upload limits](https://developers.cloudflare.com/r2/platform/limits/).
