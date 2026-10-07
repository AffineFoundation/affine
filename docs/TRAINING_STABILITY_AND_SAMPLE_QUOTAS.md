# Training stability and per-task sample quotas

Planning status, 2026-10-07: proposed experiment; production remains K=1, L=1.
This document supersedes historical checkpoint-specific launch conditions in
K2L2_QUOTA_QUALIFICATION.md, not its scientific or proof requirements.

## Establish the baseline first

The replacement trainer completed epoch 42, advancing the authentic optimizer
lineage from step 32 to 33. Independent readback verified all 23 state objects;
epoch 43 then opened automatically. Epoch 43 subsequently completed naturally
in 42.41 minutes, advanced optimizer step 33 to 34, and published its checkpoint;
epoch 44 opened automatically. The epoch 42 training job took 20.62 minutes,
which is not the duration of the entire epoch. Continue observing ordinary epochs
before changing production quotas. Do not interpret recovery downtime as
instability caused by a two-rollout task batch.

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

The original checkpoint-32 evaluation subsequently completed all 24 groups and
750 tasks, with actual original process waits of zero. Independent reconstruction
verified the original signed jobs and reports, durable readbacks, and the same
task identities, prompts, seeds, harness, source and runtime across all three
models. Results were base 548/750 (73.07%), checkpoint 24 482/750 (64.27%), and
checkpoint 32 496/750 (66.13%). Checkpoint 32 remains 6.93 percentage points below
base; its paired bootstrap 95% interval is [-10.13, -3.60] percentage points.
Its 1.87-point increase over checkpoint 24 has interval [-1.20, +4.80], so this
comparison does not establish recovery or convergence. These are exploratory,
unadjusted intervals from one fixed native-evaluation cohort, not TOPLOC proof
assurance. The independently reviewed comparison receipt SHA256 is
`0516ffa3b05683af6d98424c6f01c3c5db3a167d73f123fb45c92c4613a08523`.
This finding strengthens the need to test objective and negative quality before
changing sample quotas; it does not identify the cause or evaluate later
checkpoints. Production K1/L1 remains unchanged.

## Recorded update diagnostics

A local reconstruction of epoch 43's original selected documents matched all
256 document hashes and actual training-pair identities. Of the 256 claimed
negative trajectories, 109 (42.6%) were exactly 1,024 tokens with no EOS and
147 ended in EOS. Of the positives, 255 ended in EOS and one was at the cap.
Mean output lengths were 488.1 tokens for negatives and 120.2 for positives.
These are authenticated unaudited training inputs, not proof-verified outcomes.
Reaching the cap without EOS is a cap indicator, not an independently recorded
stop reason, and every claimed `done` flag being true does not establish EOS.
This makes negative completion/length quality a concrete hypothesis to test
before treating larger per-task quotas as the fix; it does not establish cause.
At the subsequent audit cutoff, all 256 exact selected batch hashes had
authenticated accepted reports under the deployed audit contract: 109/109 in
the capped-negative subset and 147/147 in the EOS-ending subset. No confirmed
invalid, ambiguous or infrastructure outcome appeared in that matched subset.
Training consumed them as unaudited inputs; the later audit join is separate
evidence and does not imply training waited for those reports. The cap finding
therefore concerns learning quality rather than rejected model computations.

A read-only inspection of the nine ordinary reports advancing optimizer steps
23 to 32 found 256 tasks and 256 pairs per update. Recorded gradient norms ranged
from 0.064 to 0.228, below the clipping threshold of 1.0, with no nonfinite norms
or margins in those reports. Mean training-pair margins increased in each update
(about 0.053 to 0.141). These diagnostics show the preference objective changing
as intended on its inputs; they do not establish held-out improvement. They do
not support explaining the observed regression as an optimizer batch of two or
as exploding gradients in these nine recorded updates. Compare negative quality,
objective and generalization before raising sample quotas.

## Completed-negative comparison

The original same-parent negative-quality experiment completed on 2026-10-07.
Both arms made one update from checkpoint 24 and Adam step 24, with eight matched
training tasks and common positive trajectories. The completed-negative arm
scored 85/128 on the same held-out task/prompt/seed cohort, versus 77/128 for
the capped-negative arm: eight gains and no losses. The paired difference was
6.25 percentage points, with a 20,000-draw paired bootstrap 95% interval of
[2.34375, 10.9375] points and exact two-sided McNemar p=0.0078125.

ROOT independently authenticated all eight original successful process waits,
the original signed jobs and full report hashes, both branch model archives,
and the complete 67-member metadata archive after durable full readback. This
supports the completed-negative package in this one-update comparison. Length,
termination and negative content changed together, so it does not isolate an
EOS effect. The diagnostic cohort has been inspected repeatedly; this is native
grading, not new inference-proof assurance, convergence or a quota qualification.
Keep production unchanged while testing the objective separately and confirming
learning on wider prospective held-out evaluations.

## Objective hypothesis to test separately

The current preference loss is `-logsigmoid(beta * (margin - reference))`.
The reference margins are recomputed from the input checkpoint of each epoch;
production currently makes one optimizer update in that epoch. This explains
why the recorded pre-update loss is approximately log(2) each time: the current
margin initially equals the newly captured reference. It is not evidence that
the optimizer failed to update. It also means there is no single fixed reference
margin carried across epochs in this objective. Whether this repeated local
contrastive update generalizes is a separate question from samples per task.

After the negative-quality comparison, compare the existing objective against
one explicitly chosen alternative from the same model and optimizer parent,
with matched data, task weights, update count and held-out settings. Inspect
positive and negative log-probabilities separately, not only their difference;
an improved margin alone does not establish improved solving ability. Keep any
objective experiment separate from a quota change and declare any changed
reference or regularization settings before launch. Do not change production
based solely on the loss value or this code-level hypothesis.

At an exactly matched epoch-start reference, the scalar derivatives are
`d(loss)/d(mean_logprob_positive) = -beta/2` and
`d(loss)/d(mean_logprob_negative) = +beta/2`. A CPU autograd check of the shared
loss confirmed -0.05/+0.05 at beta=0.1 for three different margins. This is a
first-update local-gradient statement, not a general equivalence after updates,
and it does not diagnose generalization. The next objective candidate, after
the negative-quality study, is a positive-only ablation weighted by beta/2 to
preserve the positive coefficient while removing the negative term. It must
use the same parent model/Adam, tasks, positive trajectories, hyperparameters
and update count as its baseline. It is research-only, not a deployed change.

Restoring the same Adam parent preserves historical first and second moments;
the positive-only arm removes the current negative-gradient term, not the
influence of past negative updates. Weight decay also remains unchanged. Match
the clipping rule and record pre-clip norms: removing a gradient component can
change the total norm and therefore the scale applied by clipping. The matched
positive coefficient describes the loss derivative before clipping and Adam,
not equality of the final parameter update. These are interpretation limits,
not reasons to reset the optimizer or silently change the experiment.

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

The initial experiment should keep production unchanged. Use a nested attempt
stream so the K1/L1 arm uses the first qualifying success and failure, while
K2/L2 adds the next distinct qualifying success and failure. Record collection
cost and completion probability for every attempted task, including tasks that
cannot supply two distinct successes. Compare equal-task training first, then
equal-wall-time supply: these answer different questions. Evaluate on the same
750-task held-out cohort at baseline and after each update; keep that cohort
excluded from mining and report paired uncertainty. Run at least three matched
updates before considering a bounded production pilot. Treat this as an initial
evidence threshold, not a claim that three updates prove convergence. Before
dispatch, freeze the experiment's seeds, sampling budget, learning settings,
numerical failure criteria and acceptable throughput tradeoff.
After experimental arms diverge, further on-policy updates require each arm's
own checkpoint-bound trajectories. Common random draws/tasks can remain matched,
but one arm's tokens must not be relabeled as generated by the other checkpoint.
Repeated updates on a single frozen parent dataset would instead be an offline
replay experiment and must be labeled as such.

## Prevent duplicate credit and duplicate training

Implementation readiness, 2026-10-07: the live paths already parameterize K/L,
reject repeated canonical prompt/output content within a submitted task batch,
construct disjoint pairs with `zip(positives, negatives)`, and average their
losses within each task. Targeted CPU admission tests cover four distinct
rollouts, reused content with changed attempt metadata, and content reused under
the other claimed class. These tests establish cheap admission behavior only;
they do not establish inference validity or production K2/L2 readiness.

The stronger cumulative-slot and application-ledger components currently live
in default-off research modules `ops/paired_quota_batch_adapter.py` and
`ops/paired_quota_research_ledger.py`. They are not yet integrated into live
submission/training paths. Existing optimizer pair identity hashes the complete
rollout, including metadata; it is not a global canonical-content dedup key.
Signed job/parent/namespace recovery protections are not equivalent to a
cross-job consumed-input ledger. Integrate and qualify these protections before
activating a higher-quota contract; do not describe them as already deployed.

Integration order: first extract and test a pinned shared identity module;
then integrate transactional submission slots and frozen revisions; then bind
learner selection and trainer applications to those identities; finally thread
the new signed policy through miner, continuous auditor, scoring and public
contract documentation. Absence of that policy preserves historical admission
and original source inventories. Do not activate it midway through an epoch.

The first extraction is now implemented as `subnet/trajectory_identity.py`:
the committed learner and research cumulative checker share the same ordered
prompt/output digest. Golden-byte and metadata-repacking controls preserve the
existing digest exactly; ROOT independently reran 98 admission, quota, ledger
and identity controls successfully. This source change does not replace the
trainer's full-pair identity, add a durable submission journal, establish
exactly-once optimizer publication, or activate higher quotas. Frozen deployed
source and signed historical jobs retain their original bytes. Next, integrate
durable authenticated cumulative revisions and recomputed execution/content
identities before considering a higher-quota contract.

Submission ownership follows the authenticated hotkey, while the current
prescribed random-draw context does not include a hotkey. Keep those bindings
separate and do not silently change the sampler when adding slot identities.
The learner still pairs authenticated claimed classes after cheap checks;
native grading and inference confirmation remain continuous audit work. Neither
the identity ledger nor a quota change may turn those confirmations into a
training prerequisite. Cumulative submission updates add immutable completed
task records; identical redelivery is idempotent. If partial task revisions are
introduced, their permitted additions and freeze rules need an explicit contract
rather than assuming an ordinary presigned PUT is a transactional append.

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

Research ledger boundary controls also reject moving the same checkpoint-bound
execution into a new epoch, while allowing identities bound to a genuinely new
checkpoint to have the same answer text. The latter is cheap identity admission,
not evidence that the new checkpoint generated those tokens; inference audits
still establish that separate claim. These controls are not live-ledger activation.

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
