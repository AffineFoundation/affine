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

The authenticated E38 training report advances optimizer step 28 to 29 with
256 task pairs. Its gradient norm before clipping is 0.1088867 against a 1.0
limit; all 256 recorded before/after margins and recomputed deltas are finite.
Margins increase for 233 pairs, with a mean change of 0.14066 nats per token.
The reconstructed post-update preference loss is 0.68622 versus an initial
0.69315. These are training diagnostics, not held-out improvement or proof of
stable learning. Ninety-two initial margins exceed 5 nats; margins alone do not
establish capped outputs or fraudulent sampling in those E38 inputs.

The deployed objective uses an immutable copy of the current epoch's input
checkpoint as its reference, then refreshes that reference for the next epoch.
Consequently an initial loss near log(2) is expected; its repetition is not
evidence that the optimizer was reset. This also does not supply a persistent
anchor to the original base model. Keep this reference rule identical in the
K1/L1 versus K2/L2 comparison. If matched held-out regression persists, study
reference anchoring or explicit drift control in a separate experiment rather
than changing the reference, quota and learning rate together.

The latest authenticated same-128-task comparison scores checkpoint 29 at
76/128 versus checkpoint 28 at 78/128: four gains and six losses on matched
tasks, prompts and seeds. Checkpoint 23 remains higher at 90/128. Successful
epoch execution therefore does not establish convergence.

The partial untouched-base comparison raises a regression concern. It does not
establish that sample count caused it. Capped negative trajectories, termination
behavior, objective/reference behavior and task-selection bias remain candidate
causes. Complete the existing matched evaluations and same-parent negative
control before attributing a learning change to quota alone.

CPU inspection of the original E37 inputs identifies a distinct failure-tail
mode. Of 256 native-accepted training pairs, 91 (35.5%) have reference margins
above 5 nats, averaging 10.91; the other 165 average 0.065. These are positive
minus negative **mean log probabilities per output token**, not sequence sums.
All 91 high-margin negatives reach the 1,024-token cap without EOS. They average
95.8% distinct token IDs and 9.88 bits of empirical token entropy, compared with
5.09 bits for the low-margin negatives; they are not empty or dominated by
repeated tokens. Only three low-margin negatives reach the cap.

The exact completed audits accepted 89 of those high-margin batches and rejected
two for TOPLOC. The native filter and trainer use different pair-digest schemas;
the comparison instead authenticates original document and batch hashes,
recomputes both pair identities, and checks the original prompt/output bindings.
For all 89 accepted high-margin batches, small original probability artifacts
give a mean FP32 claimed margin of 10.915 versus the trainer's BF16 reference
mean of 10.902. A BF16-only explanation therefore does not explain this mode.
The signed audit contexts use the same checkpoint and plain autoregressive
harness, temperature 0.8, top-p 1 and cap 1,024. Their frozen verifier requires
selected-logprob comparison against recomputed logits and prescribed CDF checks,
with exact cached replay for calibrated support/boundary adjudication. Historical
reports do not retain per-position CDF results or replay counters, so this source
and receipt review is not an independent replay of those GPU computations.

The original selected-logprob traces usually change after a prefix: the median
first position below -5 nats is 50, and the median onset of a sustained
low-probability tail is 53. The final 512 tokens average -11.856 nats per token,
near the log of the 152,064-entry model vocabulary. In 82 of the 89 accepted
traces, tokens outside the tokenizer's 151,665-entry mapping occur only after
that onset. This suggests a post-prefix failure mode to investigate; it does not
establish its cause or classify a capped failure as fraud. It also does not show
that simply increasing sample quota will fix training stability.

Complete the existing matched same-parent capped-negative versus completed-EOS
negative control before changing quota. Keep the original positive trajectories,
parent model and optimizer, tasks, update count and held-out evaluation settings
fixed, and report cap/EOS rates, failure-tail diagnostics and paired held-out
gains/losses. Keep the original full held-out comparison running independently.
These observations activate no production admission, sampler or quota change.

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
