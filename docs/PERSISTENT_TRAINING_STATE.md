
## Bounded idempotent state PUT retries

Future worker sources retry only HTTP408/429/500/502/503/504 and transient
connection/timeouts, at most four attempts with bounded exponential backoff.
Each attempt rewinds the same open regular-file descriptor and verifies unchanged
inode/size/mtime/ctime. Authorization errors, redirects and permanent responses
fail immediately; grants are never refreshed implicitly. Final diagnostics retain
the HTTP status/attempt count or a sanitized transport-exhaustion category, never
a signed URL or response body. This code does not alter any sealed running source.

An export failure after a training update is not a pre-update failure. Partial
optimizer shards and a BF16 checkpoint cannot reconstruct missing FP32 masters
and AdamW moments. Preserve the failed original and its evidence; a recovery
requires an explicitly authorized distinct attempt from the last complete durable
parent, with its own output namespace. It must not claim the failed original
completed or synthesize a missing durable optimizer step.
