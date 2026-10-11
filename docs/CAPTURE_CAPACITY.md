# Coordinator capture capacity

The `bounded-parallel-token-capture-v3` policy is prepared for a future signed
epoch. It is inactive until the operator publishes that policy in the epoch
manifest. Existing miners need no upgrade for this capacity change.

It raises coordinator token-document capture from 16 to 64 concurrent workers.
The total freeze budget remains 60 seconds, including commitment discovery and
authentication; this does not guarantee 60 seconds of token downloads. Each
worker reads at most 2,000,000 document bytes plus one oversize sentinel, giving
128,000,000 raw document bytes plus 64 sentinel bytes in flight. Decoded objects,
HTTP buffers and the existing gateway state require additional memory.

The same bounded connection pool handles GET and immutable PUT operations.
Per-request connect/read timeouts are 5/10 seconds, with one SDK attempt; the
capture loop owns retries and reconciles uncertain publications through its
fsynced journal. Parent commitment publication remains at 16 workers. Both the
small commitment journal and token-document journal remain durable.

The postcommit random seed, rank ordering and cutoff rules are unchanged. A
worker takes the next ranked document when a slot is free; eventual inclusion
still depends on completion before the deadline. A transport failure or cutoff
is infrastructure deferral, not evidence of cheating. Unvalidated duplicate task
claims cannot by themselves revoke another miner's reward. Historical v1/v2
capture rules and signed epochs retain their original interpretation.

Proof verification, K/L quotas, the per-miner batch cap, attempt nonces, mining
deadlines, training eligibility, optimizer lineage and hourly weight submission
are unchanged. The pinned miner source can remain the same because capture
metadata is coordinator-only. An already-issued epoch must not be retrofitted
with v3: activation chooses the first unopened epoch and preserves historical
completion/reentry paths.

The implementation is in `subnet/training_documents.py`,
`subnet/commitment_transport.py`, `Bucket.capture_io_client` in
`subnet/storage.py`, and the prospective operator helper
`ops/trainer_lifecycle/capture_capacity.py`. The matching operator admission entry
is `ops/trainer_lifecycle/capture_capacity_entry.py`; private deployment policy and
parent runtime references are deliberately external to the repository. It verifies
the authenticated predecessor, amends only its capacity reentry guard, and retains
physical discovery/config bindings. The operator's signed source policy
must bind the installed implementation and preserve its current parent service
and singleton. The public source also retains its separately gated token-artifact
transport support; this capture change does not activate that transport.

For the first completed v3 epoch, inspect attempted/captured/deferred documents,
structural failures, peak in-flight work and the new `stage_timing_seconds` fields:
GET/body, validation, WAL intent, immutable PUT, WAL commit and global checkpoint.
These are accumulated wall times and maxima; concurrent sums need not equal
elapsed time. The last checkpoint's timing is recorded by the next normal state
persistence. Compare real throughput before claiming a speedup.

CPU tests cover 64 simultaneous requests, original rank ordering, deadline and
failure handling, uncertain PUT recovery, restart without refetch, and actual
Gateway capability/commitment flow through both journals. They use local transport
fixtures, not a live R2 throughput benchmark.
