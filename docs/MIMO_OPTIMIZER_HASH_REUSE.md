# Transaction-scoped staging hash reuse

This optimization is prospective: it requires a newly qualified source release.
It does not change optimizer equations, moments, checkpoint lineage, sampler,
training inputs, publication content, or model-only upload policy.

## Repeated work

Originally each optimizer shard was fully hashed three times in a successful
fresh update: after `CPUAdamW.save_local` wrote it, while staging assembled its
local-state inventory, and when the transaction built its artifact inventory.
The transaction already cached its own second inventory check, but that cache
did not include the two earlier reads. With approximately 107GB of local state,
the two extra passes can read over 200GB per update.

The transaction now offers an optional `stage_with_hasher(report, root,
hash_artifact)` callback. The established two-argument `stage` API remains
available; exactly one callback protocol must be supplied. The trainer service
uses the new callback and passes its hash function through staging to
`save_local`.

## Integrity boundary

`hash_artifact` always obtains its first digest from an actual complete, guarded
file read after writing. It accepts only artifact paths inside the original
transaction's model/optimizer directories. It cannot accept descriptor-supplied
hashes, inventories or serialized cache records. Successful reads seed the same
in-memory memo already used by transaction inventory validation.

Reuse retains the existing ordinary-path, no-symlink, inode, size, mtime, ctime,
open-descriptor and fsync checks. Changed stamps require a fresh full read. A
changed digest after the original staging hash aborts rather than silently
adopting the modified artifact. Replacing a file with identical bytes still
requires a full read before it can pass.

The function is valid only inside the exclusive staging callback, in its original
process/thread. It expires on normal return or exception. Every transaction run,
including a retry on the same Python object, starts with an empty memo. No memo
or receipt is persisted. Cold recovery and lost-publication-ack retries therefore
rehash every staged artifact before publication and do not execute another
optimizer update. The host's sole trainer lease and transaction lock remain
mandatory.

As with the existing memo, filesystem stamps plus exclusive ownership are a
coherence optimization, not proof against a malicious concurrent same-UID writer
who bypasses ownership and writes within a shared timestamp tick. No stronger
security property is claimed.

## Validation

Tests cover actual CPU AdamW save/restore with exact master/moment equality,
identical transaction journals, mutation with restored mtime, inode replacement,
file/ancestor symlinks, out-of-root and cross-thread calls, expired callbacks,
interrupted staging, same-object and new-object publication retries, a fresh
subprocess recovery, and tampered recovery artifacts. Existing training,
transaction integrity and worker tests also pass.

`tests/bench_mimo_staging_hash_reuse.py` is a bounded local I/O benchmark. Three
64MiB optimizer-like runs produced identical journals while reducing streamed
hash input from 192MiB to 64MiB (plus tiny model metadata). Median write/hash/
transaction time was 0.249s before and 0.150s with reuse on this host. This is a
microbenchmark, not a production speed guarantee; optimizer serialization,
model upload and storage bandwidth still contribute to epoch duration.

Run the benchmark from the repository root with `PYTHONPATH=.:tests` and:

```sh
python -B tests/bench_mimo_staging_hash_reuse.py --mib 64 --rounds 3
```
