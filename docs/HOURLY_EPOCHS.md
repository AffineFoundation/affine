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

Verified status at 2026-10-05 03:30 UTC: original epoch 10 completed its three
training updates, committed all 23 FP32 optimizer shards (91,387,491,264 bytes),
and published checkpoint `6a2bb631…`. Its original after-training evaluation
completed at 21/32, compared with 19/32 on the same pre-training cohort. This
small diagnostic does not establish a trend. The controller is idle at round
eleven with the same committed learned parent; it has not opened a successor. Four verifiers are
admitted. One replacement H200 is running its original isolated qualification control;
the other terminated with a GPU error and remains unenrolled. Neither is serving
normal audit jobs. The two earlier reserve machines were deleted by the
legacy reaper; retained replacement protection is now explicit.

The old audit budget of 256 was larger than the submitted population, so its
nominally sampled policy effectively checked every pair. Authenticated original
job measurements give a mean cost of 80.76 verifier-seconds per selected pair
and a job-cost p90 of 132.16 seconds per pair. These measurements do not establish
the throughput of six workers running the new source.

The next preparation uses 12 initial checks and four reserved escalation checks
for the four qualified workers, or 18 plus six after six workers qualify,
with a hard 600-second audit window. Minimum allocation is zero when capacity
cannot cover every identity; maximum remains three per identity. Escalation
charges repeated checks as well. Only completed accepted audits earn points or
feed training. Unallocated or unfinished batches are not fraud. One confirmed
invalid audit would zero the miner's epoch points under this prospective policy.

The proposed phase budget is 600 seconds mining, 300 freezing, 600 auditing,
1,200 training/publication, 300 weight handoff, and 600 slack. One configured
update still covers its admitted training cohort: reducing three updates to one
does not itself reduce the number of task forwards. The smaller audit cohort,
parallel state publication/readback, and independent checkpoint evaluation must
be measured together before asserting a one-hour epoch.

These budgets and penalties are preparations, not the active contract.
Small-commitment sampled auditing and independent evaluation still require the
genuine GPU control, source admission, and safe learned-parent cutover.
The published epoch manifest and llms.txt govern external miners until a
prospective source and contract are explicitly activated.

Release evidence must include several consecutive real epochs within the
hour, authentic chain transactions, real monotonic optimizer updates and
durable publications, external-miner compatibility, and comparable held-out
results. Passing unit tests alone does not establish throughput or learning.
