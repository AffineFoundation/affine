# Prospective sole-current FP32 optimizer cache

This feature is **off by default and not deployed**. Enable it only with a new
signed epoch manifest containing:

```json
{"optimizer_state_local_cache":{"version":"sole-current-fp32-state-cache-v1","max_checkpoint_bytes":137438953472}}
```

It also requires qualified independent full state publication and a source pin
for `subnet/optimizer_state_cache.py`. The public/private manifest projections,
runtime source qualification and post-ACK cleanup helper must all include the
new module before activation. It changes local optimizer transport, not the
miner's sampler or the requirement for complete durable R2 publication.

An uploaded FP32 shard moves into one bounded candidate cache after its normal
hashing and successful PUT. That candidate is not usable as a parent. The
independent reader still reads all durable shards and verifies every SHA and
size; ROOT commits the optimizer lineage normally. Only the exact signed
post-durability ROOT acknowledgement can promote the original job's candidate.
Promotion binds the actual report, descriptor, checkpoint, counter, namespace,
source and original job, and rechecks all local candidate hashes and sizes.

On the next original job, the trainer verifies the exact approved parent
descriptor and original signed source, file ownership and every shard SHA/size.
It then consumes cached shards into the ordinary restore workspace. Existing
restore code still performs complete hashing, tensor/schema, finite-value and
disjoint-coverage checks before retiring each restored shard. The consumed
parent disappears before the new candidate is exported, preventing overlap
between two complete historical state caches. A cache lease prevents concurrent
jobs and promotion from using or retiring the same state.

Missing or corrupt owned cache bytes trigger the normal R2 restore with evidence
of the cold path. Replaced inodes, symlinks and unowned members fail closed.
An interrupted, unpromoted candidate can be retired when a different original
ROOT job begins; the same original job must recover its staged completion
instead of silently retraining. A stale parent cannot discard a newer committed
cache. Durable R2 state and historical reports are retained in every case.

Capacity admission includes retained state, bounded concurrent transfer space,
the complete BF16 export and existing disk/RAM reserves. Existing parent bytes
receive reclaim credit only after full binding, SHA and size verification,
because they are already occupying disk and will be consumed before export.
The ordinary resource admission remains independently enforced. At the observed
roughly 160 GB free on the current trainer, a 91.4 GB retained state plus the
existing transfer/export/reserve budget leaves little headroom; a real capacity
probe is required before qualification or activation.

Tiny real FP32 optimizer controls show identical cached/cold restored masters,
moments, counters and subsequent AdamW updates. They cover unpromoted candidates,
corruption, missing files, incorrect sources, signed ACK failures, leases,
ownership guards, capacity and stale lineage. No full-size GPU timing or epoch
speedup has been measured. The intended traffic reduction is one complete
parent-state GET per warm job; export and independent full readback remain.

Post-ACK activation pins and loads `optimizer_state_cache.py` explicitly alongside
both lifecycle helpers. Cache-enabled GPU execution additionally authenticates
`cache_lifecycle.py`; policy admission before source authentication is pure and
does not import either cache implementation. Both execution modules are loaded
through the authenticated fresh-source finder.

Cache-enabled cleanup uses one detached CPU supervisor identified by the exact
original job and signed ACK hash. The launch intent is durable before launching;
a lost SSH response only causes observation of that original handle. The helper
has a 30-minute execution bound and short SSH status probes. A terminal failure
or unknown launch outcome requires operator recovery; it never triggers a second
trainer execution. A signed promotion intent prevents the next trainer from
consuming or abandoning pending state until promotion completes. Exact confirmed
`current.json` promotion is idempotent, including a crash immediately before
`pending.json` removal, and never rehashes shards on a successful ACK retry.

Dispatch checks promotion status before creating the next original trainer job.
The coordinator synchronously persists the genuine original post-commit ACK
intent before returning from training completion, then schedules slow cleanup
and promotion in the background. A lost preparation reply or controller exit
recovers that same intent; it cannot advance training first. An original pending
candidate with no promotion guard also blocks dispatch, and a worker never
abandons a candidate matching its actual approved parent while the ACK is absent.
This closes the interval before a delayed cleanup thread starts its supervisor.
If promotion starts during worker startup, the worker releases its cache lease
while waiting boundedly, allowing the original helper to complete, then continues
the same training job. Confirmed terminal promotion failure affects only the
optional byte cache: the worker authenticates the original signed ACK, job,
report and exact approved parent descriptor, preserves those records and failed
candidate inventories, retires only the owned cache bytes, and restores from R2.
Unknown launch outcomes or a matching live child never authorize retirement or
cold fallback. Terminal evidence includes child PID, process start identity and
confirmed process exit; a fast child that exited before identity capture must
also have a confirmed exit code and no extant process at that PID.
A terminal failed promotion is consumed exactly once under the cache lease: its
full signed failure record is archived before the active guard is removed. The
old ACK consequently cannot block a later approved parent after training advances.
