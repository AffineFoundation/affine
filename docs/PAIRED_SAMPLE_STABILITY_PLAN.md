# Larger per-task batches: proposed stability experiment

This is a plan, not an activated miner contract. Production remains K1/L1.
Do not change old manifests, sampler contexts, receipts, rewards or optimizer
lineage. Existing production and evaluation jobs continue independently.

## Question and current evidence

One success and one failure per task may give a noisy preference estimate, but
the optimizer update can already average 256 distinct tasks. It is not a
two-trajectory optimizer minibatch. The task-normalized trainer averages pair
losses within a task, then task losses within the update. Larger per-task groups
must retain that normalization, rather than silently increase a task's weight.

The partial untouched-base comparison raises a regression concern. It does not
establish that sample count caused it. Capped negative trajectories, termination
behavior, objective/reference behavior and task-selection bias remain candidate
causes. Complete the existing matched evaluations and same-parent negative
control before attributing a learning change to quota alone.

## Proposed contract and duplicate identities

Start with K2/L2: two distinct successes and two distinct failures for one task,
on one pinned checkpoint. Require four distinct prescribed attempts and four
distinct trajectory contents. Different attempts can legitimately produce the
same output; these repetitions do not fill extra quota and are not fraud solely
because they repeat. Existing proof, sampler and native grading checks remain.

Use three distinct keys, centrally recomputed rather than trusted miner IDs:

- Task slot: epoch, checkpoint, versioned taskset, environment index and miner
  identity. One cumulative submission slot per task; updating its contents does
  not create another reward entry.
- Execution identity: approved sampling context, task, harness and attempt.
  Repeating an attempt cannot create another sample. Preserve the current draw
  binding; adding a miner-dependent sampler would be a separate contract change.
- Trajectory-content identity: checkpoint, versioned task, harness and canonical
  full prompt/output/action/observation trace. Exclude upload timestamps, claimed
  labels/rewards, proof serialization, filenames and miner identity. Repacking
  or relabeling a trace cannot make it new. Keep execution identity separate so
  legitimate repeated output can be recognized without a fraud accusation.

Check these keys across the entire committed cumulative submission, not merely
inside one file. Preserve cross-miner same-task duplicate-zero scoring. Content
identity must also work across identities so copied traces cannot become new
by changing a UID. Historical checkpoint traces cannot fill a new checkpoint's
quota; hashes and prescribed draws must bind the actual new checkpoint.

Freeze a selected task-slot revision once for training. Persist its content IDs
and training-job association so replayed uploads, retries or later cumulative
revisions cannot cause a second application of the same selected data. A retry
of an already completed job returns its existing receipt, not another update.

## Training and reward semantics

Use two nonoverlapping positive/negative pairs per task in deterministic order;
do not reuse one success in multiple pairs or create a four-pair cross product.
Each pair gets weight 1/(2T), retaining total task weight 1/T for T selected tasks.
Cheap eligibility still admits unaudited training inputs; continuous audits run
independently. One qualifying unique task still earns one contribution unit,
scaled by the existing validity estimate and penalty policy, not two pair points.
Count each committed trajectory once as audit evidence; redelivery cannot inflate
confidence. Keep independent training-input deduplication and reward deduplication.

## Controlled qualification and rollout

1. Add adversarial controls for repeated uploads, changed filenames/proof bytes,
   reordered samples, changed claimed labels, repeated attempts with different
   claims, repeated output from different attempts, cross-UID copies, old-model
   replay, reused pair members, and trainer restart/recovery double application.
2. Collect a common prescribed attempt stream on a fixed parent under one fresh
   research sampling context. Compare nested K1/L1 and K2/L2 selections from the
   same tasks; do not let quota-specific contexts change the compared draws.
   Use the existing attempt budget first and record tasks unable to reach K2/L2.
3. Fork the same model and persistent optimizer into two isolated branches. Keep
   task count, learning rate, objective/reference, token limits and update count
   fixed; vary only one versus two nonoverlapping pairs per task. Measure coverage,
   gradient norms, KL/reference drift, cap/EOS rates and actual GPU/transfer cost.
4. Run matched held-out evaluation with the same tasks, prompts, seeds, runtime
   and generation budget. Report paired gains/losses and uncertainty; compare
   several controlled updates, not only in-sample loss or one small diagnostic.
   Also measure useful tasks per unit mining budget: more per-task samples may
   reduce diversity and throughput even if the fixed-task comparison improves.
5. If qualified, activate a new manifest/contract at a future epoch boundary.
   Publish miner instructions and llms.txt before activation. Explicitly update
   sample/byte/audit budgets for four trajectories and preserve old-contract
   verification for historical submissions. Confirm external-client compatibility
   and an end-to-end epoch before broad rollout. Otherwise retain K1/L1 and use
   the evidence to investigate the objective or termination problem instead.

No automatic K4/L4 escalation or production learning-rate change is proposed.
