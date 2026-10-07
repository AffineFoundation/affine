# Consumed-input application boundary (default off)

This review concerns the ROOT-identified f213 scientific source and E41v7d
operator overlay selected by the signed E42 replacement policy. The private
review receipt records full file hashes. Reading those local files is not a new
physical observation of the live trainer or fresh operator-signature validation.
No production caller imports `ops.consumed_application_ledger`; K1/L1 and
unaudited training eligibility remain unchanged.

## Existing guarantees and concrete gaps

- `remote_backend.training_resume` authenticates the stored original signed job,
  manifest, steps, submission SHA/accepted batch and receipt inventory. `run`
  observes that original report/status and refuses automatic train relaunch after
  failure/absence. This is strong original-request resumption, scoped to a label.
- `persistent_training_controller.train` checks cached metrics against the exact
  original job/report, input inventory, output optimizer lineage and independent
  inference/state publication. Recovery of the same completed job promotes its
  verified output rather than executing another update.
- `persistent_publication.complete` stages/checks both model and state paths before
  descriptor-last authority publication. `_publish_verified_descriptor` rejects
  a different existing publication in that original job namespace and verifies
  readback. FP32 parent/master/moment lineage is preserved; an unchanged BF16
  model can still have a different advanced optimizer descriptor.
- `commit_latest` rejects a stale parent and accepts the same pointer idempotently.
  Its read/check/`save` sequence is not a transactional multi-controller compare
  and swap. `save` uses a temporary file plus replace without fsync. Current
  serialized orchestration must not be described as a concurrent or power-loss
  consumed-application transaction. Authority objects, local latest pointer and
  epoch metrics are separate writes.
- The operator cheap learner decoder rejects duplicate prompt/output traces
  within a document; job admission requires unique committed slots/tasks; frozen
  collection excludes multiple candidates for one task. Its immutable post-freeze
  selection prevents redraw on restart. These controls do not constitute a
  cross-recovery optimizer consumed-input journal.
- `covered_epoch_optimizer.pair_identity` hashes full positive/negative rows.
  Exact pair dedup therefore is not token-canonical wrapper/relabel dedup. A CPU
  control confirms adding UID metadata changes that pair identity while the
  shared token-trace digest stays equal. This helper observation does not prove
  that current live admission accepted such a duplicate.
- `task_normalized_training.task_groups` intentionally cycles a task when there
  are fewer tasks than prescribed steps. One task and three steps yields three
  groups containing that same task. A global content-once rule would erase
  intended optimizer updates. Mean-pair-within-task / mean-task-within-update
  weighting must remain unchanged.
- `training_startup_recovery` admits distinct signed, witnessed pre-compute,
  pre-update parent-restore, and uncommitted post-update recovery classes. The
  last class explicitly records one discarded original update and permits a
  fresh execution from the authentic durable parent with preserved inputs and
  scientific pins. That is an authorized recovery, not evidence of exactly-once
  physical execution. Existing historical declarations must remain interpretable.

## Narrow prospective guard

`ConsumedApplicationLedger` is a separate private SQLite WAL/FULL research
ledger. Creation/reopen requires `enabled=True`. A TRUSTED original-boundary
callback authenticates actual captured documents/admissions and derives the
binding; a boolean or caller-supplied binding is refused. The callback must also
pin the approved branch/cohort, actual parent descriptor and inference checkpoint,
parent counter, plan, settings/scientific source/runtime, canonical selected
revisions and exact ordered task groups. Current code supplies this storage
interface, not that live boundary implementation.

The canonical application digest excludes job labels, URLs and wrapper bytes.
Distinct authenticated original hashes become immutable aliases. Parent slot
identity is `(authorized branch, parent optimizer descriptor SHA)`: a changed
plan, counter, population, schedule or settings cannot reserve that same parent
again. Branch is trusted study authority, not a caller-selected retry escape.
Independent matched study arms must be predeclared distinct branches; all jobs
and recoveries of an arm must share its branch. No new genesis/reset is allowed.

Selected revisions reuse existing canonical execution/content identities and
reject cross-task copies. Update groups reject within-group task duplication but
preserve prescribed reuse across groups. The binding retains total per-task
contribution and pair weighting from the nested selector. The actual source
adapter must additionally authenticate/token-dedup the frozen traces through
`subnet.trajectory_identity` and reproduce original `task_groups` ordering,
reference-margin lifetime, seeds and clipping. This ledger does not derive a
valid training schedule or native grading from opaque hashes alone.

A committed claim precedes executor invocation. Competing originals/wrappers
invoke at most one cooperative executor for a parent. Exceptions or process death
leave `executing` permanently unresolved, even before any update. There is no
lease, timeout, reset, cancellation or automatic retry. A trusted publication
callback may complete the SAME application only after authentic independent full
model/state readback, exact original output lineage and expected final counter.
Synthetic tests use real Ed25519 boundary verification; they do not verify model
or optimizer bytes. A crash after output but before ledger completion can be
reconciled from genuine original publication evidence without executing again.

## Remaining integration before any activation

1. Build/version the actual controller/child admission adapter from the immutable
   captured-input journal and native selector, checking parent descriptor, token
   identities and exact scientific schedule. Do not treat selection-packet
   idempotence as application identity. Bind the original child/supervisor to the
   same canonical application and reserve before dispatch; a cooperative worker
   cannot bypass or independently recreate the claim.
2. Record evidence links for original signed job and authentic terminal outcome,
   model/state readback and authority publications. Require exact counter advance
   by the scheduled step count. Keep unchanged BF16 outputs distinct from their
   optimizer descriptor. Reconcile current pointer/metrics publication after
   crashes without new training; do not claim SQLite and object-store publication
   are one atomic transaction.
3. Specify a versioned authority-mediated recovery transition for definitive
   no-update or discarded-uncommitted candidates if needed. It must preserve all
   predecessor attempts, process absence, authentic unchanged parent and partial
   output disposition. This module deliberately does not implement such a release,
   so existing special recovery declarations cannot activate it transparently.
4. Qualify the actual durable storage, exclusive branch ownership and fail-closed
   restoration. Copying/replacing the database or choosing a new branch can bypass
   a local ledger; triggers protect its schema only. No optimizer exactly-once
   publication, distributed consensus, inference audit or learning claim follows.

CPU controls cover authenticated wrapper aliases, changed applications, declared
three-update reuse, actual original scheduling/digest helpers, concurrent aliases
and execution, signature/lineage failures, immutable unresolved claims and actual
subprocess exit78 before/after a simulated update. The simulated after-update
case fsyncs an external marker and synthetic signed output; recovery writes only
metadata. No GPU, native optimizer update, model download or deployment is run.
