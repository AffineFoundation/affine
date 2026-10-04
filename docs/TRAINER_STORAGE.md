# Training job disk admission

The prospective controller computes download space from the frozen receipts of
every miner whose audited batches enter its training request. Receipt sizes must
be positive bounded integers and match the audited submission hash; at most 256
submission files may enter a job. It counts each retained download, even if two
request entries share the same hash. Unselected uploads do not consume this
job's download reserve.

The trainer keeps those compressed ZIPs throughout the job, but decodes one
submission at a time. Admission therefore reserves the total planned compressed
bytes (at least one complete artifact budget), one raw-artifact working budget,
the existing conservative model snapshot/export allowance, any missing input
checkpoint and the fixed safety reserve. Previously the compressed reserve
covered just one submission even when the job downloaded many files.

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
