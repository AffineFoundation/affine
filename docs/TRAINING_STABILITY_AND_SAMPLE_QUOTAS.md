# Training stability and per-task sample quotas

Planning status, 2026-10-07: proposed experiment; production remains K=1, L=1.
This document supersedes historical checkpoint-specific launch conditions in
K2L2_QUOTA_QUALIFICATION.md, not its scientific or proof requirements.

## Establish the baseline first

The learner is currently held during recovery of the trainer runtime and its
checkpoint cache. Restore the authentic checkpoint and optimizer lineage and
complete successive ordinary epochs before changing production quotas. Do not
interpret recovery downtime as instability caused by a two-rollout task batch.

A submitted task batch currently contains one successful and one unsuccessful
trajectory. An optimizer update aggregates many task batches; it is not an
optimizer batch size of two. Recent held-out results are mixed. More samples per
task are a hypothesis, not a demonstrated cure. Capped, incomplete negative
trajectories are another hypothesis being tested separately.

## Wider held-out finding

The completed matched 750-task evaluation on 2026-10-07 measured 548 correct
for the base model and 482 for checkpoint 24: a loss of 66 answers, or 8.8
percentage points. The paired comparison has 113 lost successes and 47 gained
successes; its bootstrap 95% interval is -12.0 to -5.6 percentage points.
Task identities, prompts, seeds, harness, source and runtime matched. This is
native grading, not inference-proof assurance. It establishes regression for
checkpoint 24 under that evaluation; it does not identify its cause or measure
checkpoint 32. The small diagnostic subset alone was misleading as a summary
of learning. Qualify the current checkpoint on the same wider cohort and resolve
training degradation before treating increased per-task quotas as a remedy.

## Controlled comparison

1. Complete the existing same-parent comparison of capped versus completed
   unsuccessful trajectories. Keep this experiment separate from quota changes.
2. Measure supply with the same approved model, tasks, harness, sampler and ordered
   attempt stream. Compare when K1/L1 and K2/L2 fill; report tasks that never fill
   rather than dropping them. Extra samples can reduce completed task coverage.
3. Compare K1/L1 against K2/L2 from identical checkpoint and optimizer state.
   Use the same task set, learning rate, objective, reference policy and number
   of updates. Reuse the common trajectories where possible. K2/L2 requires two
   distinct successes and two distinct failures, producing two disjoint pairs,
   not four Cartesian pairs.
4. Average pair losses within each task, then average across tasks. A task retains
   total weight 1/T and earns at most one task point. Measure gradient norms,
   clipping frequency, token lengths, termination, reference margins, nonfinite
   values, task coverage, elapsed time and transfer bytes.
5. Evaluate every branch on the same held-out tasks and fixed generation settings.
   Report paired gains/losses and uncertainty. Require repeated updates and
   independent evaluation before concluding that a quota change improves learning.
   Include both equal-task and operational time-budget comparisons.

Predeclare acceptance criteria before dispatch: improved held-out outcomes over
multiple updates, no new nonfinite/gradient failures, preserved proof checks,
acceptable completed-task throughput and epoch duration. If evidence is weak,
retain K1/L1 and investigate the objective, learning rate and negative quality.

## Prevent duplicate credit and duplicate training

Use stable, authenticated identities and a durable ledger, not filenames or UID
alone. UIDs can be reassigned. Scope a task to the checkpoint, versioned taskset,
canonical task index and harness; scope ownership to the registered hotkey.

* One cumulative task batch per owner and task scope per epoch. Replacing an
  upload updates that batch, not the submission count. Frozen revisions cannot
  be changed. Enforce this with transactional uniqueness and idempotent request
  IDs so concurrent uploads and retries cannot create extra entries.
* Bind each trajectory to its approved checkpoint, task, harness, sampling
  contract, attempt index and prescribed random draw. Reject reuse of the same
  execution under another slot, batch or claimed class. Changing a filename,
  wrapper or claimed reward does not create a new execution.
* Hash canonical prompt/action/token content separately from execution metadata.
  Require distinct content within the four-trajectory task batch; different seeds
  yielding identical outputs do not provide extra sample diversity. Native
  grading determines the class. Similarity alone is not evidence of cheating.
* Apply the existing cross-miner duplicate-task rule separately from trajectory
  deduplication: duplicate valid task claims in an epoch earn zero task credit.
  Mere malformed claims must not cancel someone else's valid contribution.
  Identical answers to math questions are not, by themselves, proof of copying.
* Track frozen batch hashes and application IDs in the trainer's durable lineage.
  A retried training job must not apply the same update twice. Intentional reuse
  within a declared training schedule must be distinguished from retry replay.
  Persist consumed-input evidence with the resulting checkpoint and optimizer
  state before promoting the authoritative pointer.

## Rollout and compatibility

Implement quota fields end to end before activation: signed epoch manifest,
miner search and cumulative upload, limits, verifier quota/dedup checks, trainer
pair construction, scoring and public llms.txt. Old signed epochs remain intact.
Publish a new versioned contract and client instructions before requiring K2/L2.
Test duplicate retries, concurrent replacements, cross-slot reuse, changed
wrappers, forged outcomes, reassigned UIDs and recovery after a training crash.

Keep training decoupled from expensive audits: eligible unaudited batches remain
available to training while continuous audits inform rewards and confirmed-error
penalties. Increasing quota must not restore an epoch-wide audit barrier.
Activate a bounded prospective pilot only after the controlled comparison;
monitor held-out performance and retain a restorable pre-change model and
optimizer checkpoint in durable storage.
