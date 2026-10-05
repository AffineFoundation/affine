# One complete epoch per hour

The operational target is mining, bounded random verification, training,
durable checkpoint publication, and validator weight setting within 3,600
seconds. A phase budget is a target until measured end-to-end runs meet it.
Empty epochs and unchanged optimizer counters are not completed training steps.

The prospective submission protocol freezes small miner-signed commitments.
Each commitment binds the epoch, approved model/source, task index, and exact
artifact size and SHA256. Heavy artifacts are copied to immutable snapshots;
only randomly selected artifacts are fetched and hashed for scientific audits.
The original unpredictable challenge is persisted after commitments close.
Selected samples retain the full sampler, TOPLOC, and environment checks.

Unaudited submissions earn no verified points and cannot enter training.
Confirmed invalid audits reduce rewards under explicit contract parameters.
Network errors, verifier failures, timeouts, and exhausted audit budgets are
unresolved work, not fraud. Duplicate scoring must describe audited coverage
honestly; an unaudited claim cannot cancel another miner's verified contribution.

Independent evaluation watches committed checkpoints using fixed held-out tasks,
seeds, runtime, and comparison identifiers. Charts distinguish the latest
trained checkpoint from the latest actually evaluated checkpoint. Evaluation
does not block the next training epoch under the prospective independent mode.

Publication retains full checkpoint and optimizer-state integrity checks.
Operator state readback can stream up to four shards concurrently by default,
bounded to eight. Every shard must match its declared size and SHA256 before
the authority signs the durable descriptor. This changes transport concurrency,
not model computation, sampling tolerances, or optimizer lineage.

Current status on 2026-10-05: the original epoch 10 finished its full audit and
pre-training evaluation, and its original three-update trainer job is running.
Four verifiers are active. A fifth has passed genuine verification controls;
the sixth is undergoing qualification. Small-commitment sampled auditing and
independent evaluation are being integrated and are not the active contract.
The published epoch manifest and llms.txt govern external miners until a
prospective source and contract are explicitly activated.

Release evidence must include several consecutive real epochs within the
hour, authentic chain transactions, real monotonic optimizer updates and
durable publications, external-miner compatibility, and comparable held-out
results. Passing unit tests alone does not establish throughput or learning.
