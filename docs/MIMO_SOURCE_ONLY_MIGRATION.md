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

The lifecycle now checks a persistent ROOT-signed hold before any successor
PLAN is created. `arm_boundary_hold` takes `lifecycle.lock` and can target the
next boundary without interrupting the current epoch. At that boundary it
captures the original CURRENT bytes and finalized predecessor. The bounded
operator deadline is an alert, never an automatic release of half-migrated
state. A restarted old executable stays held even after the CURRENT migration;
only the new qualified active executable can resume after both signed ACKs.

`ops/mimo_source_migration.py` is an explicit owner-private command with `arm`,
`trainer`, `host`, and `complete` actions. It has no live path defaults, performs
no deployment, and does not write chain weights. The trainer action takes the
real `.trainer.lock`, rejects unfinished transactions, authenticates the entire
original model/optimizer inventory once, then uses the signed role journal.
The host action takes the real `lifecycle.lock` and requires the captured hold
plus trainer ACK. Completion binds the exact captured bytes and both ACKs.
Crashes after PREPARED, CURRENT or ACK resume the same transaction, never another
optimizer update. This initial operator path admits one persisted source
transition per run; replacing its completed hold needs a separately reviewed
subsequent migration rather than deleting its journal.

Historical audit/evaluation, finalized-history, dashboard and hourly-writer
readers now support the explicit registry. Original jobs retain original
policies and execution profiles. The host runner accepts an opt-in registry
and worker/source map; every worker remains on one actual installed profile.
The sole writer's composition needs its separately ROOT-signed reader config
updated to the full old/new execution registry; cursor, EMA and chain code are
unchanged. The dashboard changes are data admission only, not presentation.

## Deployment qualification still required

* Install the reviewed host-only hold hook on the running old host before
  arming a hold. Frozen v7 does not know this marker; a file alone cannot stop it.
* Qualify and archive the exact prospective source, immutable public policy and
  full changed-file allowlist. Prepare matching client archives and workers.
* Keep old-source verifier/evaluator workers for queued historical work; add
  new identities/profiles for new-source jobs. Drain the sole trainer's original
  completed job and stop its old claiming loop before the locked handover.
* Prepare the signed transition and plans from the actual boundary originals,
  not an earlier optimizer snapshot. Retain the stable trainer state root.
* Stage the historical readers and original hourly writer's signed composition
  before opening a new-policy epoch. Never replace the writer or its cursor.
* Qualify an isolated real GPU cold restore and one successor update before
  activation; verify model-only publication, original optimizer continuation,
  and historical report authentication. All production activation remains the
  root operator's responsibility; no command here has been run on live state.

## Qualification

CPU tests cover old/new exact profile routing, policy/source pin rejection,
changed model/sampler/grader/runtime/objective rejection, stale CURRENT/ACK,
quiescence and crashes after each durable trainer migration write. The actual tiny
trainer additionally completes a source-only migration, cold-restores the
original model and Adam moments, advances step 1 to 2, and retries without
advancing again. The writer regression reconstructs the same signed historical
population/report and preserves identical miner credit after registry admission. Existing
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
