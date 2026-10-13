# Source-only MiMo migration (prospective)

This is not an activated release. Keep the original epoch, trainer process,
optimizer generation, hourly writer and current public contracts running while
qualification is prepared. Never manually relabel CURRENT or rewrite an issued
manifest, job, report, receipt, or transaction.

The registry admits complete, already qualified old and new release policies.
Both retain the same run, original model, native tasks, grader, sampler,
precision, dependency runtime and training objective. Only source archive hash
and size may change in policy scope. The exact changed executable files are
listed and ROOT-signed. An archive hash is distinct from the digest of
`source_files`, which is the trainer's source closure identity.

`SourceReleaseRegistry` resolves each historical manifest to its original
qualified policy and exact executable inventory. `Coordinator` has opt-in
registry/profile routing: every worker is assigned one installed source archive.
An incompatible worker leaves a queued job untouched, without consuming an
attempt. Pending old verifier/evaluator jobs require old-source workers. A warm
worker cannot switch code in place; its existing source guard remains intact.

## Boundary transaction

1. Prepare and qualify the new source archive and registry without enabling it.
   Public qualifications must cover the exact new policy scope. Old receipts
   cannot be relabeled as new qualification evidence. Preserve current runtime,
   source archives, clients and historical policy entries.
2. Establish a persistent host boundary hold under `lifecycle.lock` after a real
   finalized epoch, before its successor PLAN/OPENING is issued. Snapshot the
   actual host CURRENT bytes, original signed opening/finalization and trainer
   CURRENT bytes. Never use the step-2 snapshot if the run has since advanced.
3. Authenticate the original successful trainer report, signed publication and
   full model readback. Validate the local published transaction/optimizer
   inventory using existing cold recovery, without running another update.
   Require the same optimizer/objective configuration. Create the signed source
   transition, retaining exact original CURRENT bytes and transaction records.
4. Under `.trainer.lock`, with no active or recoverable update, stop claiming on
   the old trainer process. `apply_role_current` writes an immutable PREPARED
   journal, atomically changes only policy/source pins, and writes the trainer
   signature ACK. Same checkpoint, optimizer step, job ID, report hashes and
   local state root are retained. Start the new executable with that same state
   root; cold restore reads the original published transaction as-is.
5. Only after authenticating the trainer ACK, migrate the host under its original
   lifecycle lock. The new checkpoint authorization refers to the original
   publication; it does not invent a new model or increment the step. Persist
   host ACK and resume opening under the new active policy/source. Preserve the
   previous discovery digest and next global round.

The supplied role mutation helper requires the caller's existing exclusive-lock
and quiescence callbacks. It is not a standalone operator command and must not
be invoked until the persistent boundary hold and consumers below are wired.
A crash after PREPARED, CURRENT, or ACK resumes idempotently: the signed journal
permits only original bytes or the exact admitted target; stale state rejects.
The host cannot migrate before a valid trainer ACK. There is no automatic
rollback after a new-policy job has been issued.

## Integration still required before activation

* Persistent hold/resume in the host lifecycle and supervised entrypoint,
  including the old host wrapper. It must survive crashes before trainer ACK;
  a lock alone does not persist a hold after process exit.
* Resolve historical audit, evaluation, finalized-history and dashboard records
  through the registry rather than the single active policy. Preserve original
  evaluation scheduler checkpoints and job IDs. Emit each receipt under its
  original policy; choose job metadata using that original source profile.
* Operator configuration/API wiring for the queue's registry and worker-profile
  map. Keep workers for unfinished old-source evaluation and audit jobs, or
  explicitly finish them before retiring those executables. Never reset leases.
* Preserve the sole hourly writer's original queue, cursor and six-hour EMA.
  Append qualified report execution profiles to its existing evidence allowlist;
  never start another writer or introduce a burn fallback.
* A reviewed boundary command that verifies actual original finalization,
  publication, trainer transaction and optimizer inventory, obtains the exact
  locks, persists the host hold, performs both role acknowledgements, verifies
  cold restoration and only then releases the next opening.

## Qualification

CPU tests cover old/new exact profile routing, policy/source pin rejection,
changed model/sampler/grader/runtime/objective rejection, stale CURRENT/ACK,
quiescence and crashes after each durable trainer migration write. Existing
queue tests must remain passing. Before production: run an isolated copied-state
cold restore and one next update with the new source; compare the unchanged
training contract, input checkpoint and counter progression, prove model-only
publication/retry, and verify historical reports still authenticate. Exact
new-policy GPU qualification receipts and worker installed-source guards remain
mandatory. This migration does not authorize v2 precision or grading changes.

## Evaluation identity across this source-only change

The current evaluation `dataset_id` hashes the suite (including seed/repeats),
native environment definition, harness, complete MiMo runtime fingerprint,
answer projection and draw-policy version. It deliberately does not hash the
source archive, release policy, epoch or checkpoint. Each draw hashes that
`dataset_id`, suite seed, task index, repeat and token position. Therefore a
source-only optimizer/journal change with identical actual sampling, runtime
and task code retains the same task prompts and draws. The CPU evaluation
regression checks actual task/prompt hashes and trajectories under archive,
policy and epoch changes, as well as equal draws at multiple positions.

A different runtime fingerprint changes `dataset_id` and the draws even if the
suite name is still `fixed32`. That must start a distinct chart series with a
fresh matched baseline. Similarly, changed sampler, task, tokenizer/template or
grader code cannot be admitted as this narrow source-only migration. Before
connecting points on the same learning chart, authenticate both original
reports and check identical dataset, suite, draw policy, runtime, harness,
projection, per-task identities and prompt hashes. Different report source
inventories are expected for this optimizer-only release, but their exact
changed-file set must be the signed, qualified source-only set; do not ignore
source differences generically. A changed scientific configuration requires a
separate baseline even when its name and task count match.
