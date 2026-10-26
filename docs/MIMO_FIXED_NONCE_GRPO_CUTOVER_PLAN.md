# MiMo reference GRPO and fixed nonce cutover plan

Status: implementation and private GPU qualification in progress on 2026-10-13;
not activated on the public network. This plan starts a fresh math training run,
replaces outcome-selected 4/4 batches with complete eight-attempt groups, and
uses the released MiMo 9B GRPO implementation. The user selected
`XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` as the starting model. Preparing this plan
does not activate a release.

The existing public contract still uses Qwen2.5-Math-7B-Instruct, four successes
and four failures selected from 1,000 possible attempts, nine batches per UID,
and task-normalized preference training. The complete-group candidates have
private integration and CPU evidence; they are not the public protocol.

## Starting checkpoint

Keep the existing math snapshot, task identifiers, mining split and held-out
split. A new model changes the tokenizer, template and numerical profile, not
which held-out questions miners may access. Pin an immutable revision of
`XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` and publish its file hashes before the
new run opens. Keep the original Qwen2.5-Math checkpoint as an experimental
control rather than continuing its model or optimizer state into this run.

The selected checkpoint already received MiMo's agentic SFT. The initial plan
starts GRPO from those published weights, with no extra Affine SFT stage.
Its architecture is `Qwen3_5ForConditionalGeneration`, using the Qwen3.5 hybrid
linear-attention/full-attention backbone and vision components. It is not the
large MiMo Pro or Flash MoE architecture. Use the actual checkpoint architecture,
tokenizer, MiMo chat template and reasoning conventions throughout generation,
verification, training and evaluation. Math inputs remain text-only.

Qualify recurrent-state handling, generation-versus-prefill consistency, backward
support, termination and the sampler on the selected runtime. Establish accuracy,
completed-answer rate, output lengths, eight-attempt mixed-group yield and stage
throughput under our actual math protocol. Published agent benchmarks do not
establish its math performance here. Preserve an untouched confirmation set for
subsequent improvement claims.

The model config specifies FP32 recurrent state alongside BF16 weights. Its
generation config permits EOS IDs `[248046, 248044]`, whereas the text config
contains the scalar `248044`. The prepared runtime now pins the two-token stop
set consistently across generation, verification, training and evaluation.
Explicitly override the packaged generation temperature of 0.6 with the chosen
reference RL temperature of 1.0. Pin thinking mode and the exact resulting
prompt tokens; neither generation defaults nor chat templates are implicit.

Sources: [selected model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B)
and [architecture configuration](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/blob/main/config.json),
plus [generation configuration](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/blob/main/generation_config.json).

## Mining and admission contract

Publish one group size, initially N=8, with independent minima of one completed
correct and one completed incorrect answer. Every task group contains exactly
the attempts numbered 0 through 7, serialized in order. Parallel generation is
allowed. Nonce ordering establishes the required set, not historical execution
order. Miners may search different tasks, but cannot replace an unwanted member
of a task's group or start another group for that same identity and challenge.

Bind random draws to the protocol, run, epoch randomness, immutable checkpoint,
runtime, exact prompt and task identity, group size, registered miner identity,
nonce and token position. Reject alternative salts, offset ranges, missing or
duplicate nonces and conflicting retries. A stable group credit ID makes uploads
idempotent across slots and retries. Identical output text from two independently
valid nonce attempts remains a legitimate sampling outcome.

Retain every attempt, including unresolved budget-capped outputs. Unresolved
does not meet the completed-incorrect minimum. An infrastructure or grader error
defers the group; it does not become a wrong answer. Preserve nine groups per UID
per epoch initially, the existing cross-miner task-collision rules, and one unit
of eligible contribution per qualifying task group. More successes within a
group do not independently multiply credit.

Keep structural admission and training eligibility separate from continuous
audits. The trainer may consume eligible unaudited submissions under the existing
policy. Audit reports feed the existing assessment and penalty mechanism; this
cutover does not introduce a verification barrier for every epoch.

Task search and mixed-group filtering deliberately condition the task
distribution. Complete groups remove within-task outcome cherry-picking, but do
not prove unbiased task selection or prohibit private extra computation. Record
coverage and the concentration of training tasks rather than making that claim.

## Reference trainer and declared adaptations

Pin XiaomiMiMo/verl commit
`a2ad9f6160b03ff2d47e59832bfb6b289f37c917` and its dependencies. Reuse its
actual loss, advantage and aggregation functions and supported optimizer path.
Wrap them in Affine's admission and update lifecycle. Avoid importing the whole
coding environment stack or relabeling our existing preference loss as GRPO.

| Setting | Proposed reference baseline |
| --- | --- |
| Advantage | Reward minus group mean, without group-standard-deviation division |
| PPO loss | Tokenwise ratio clipping at 0.8–1.2; negative-advantage dual clip 3.0 |
| Aggregation | Sum active token losses within a task group, divide by that group's active token count, then average task groups |
| PPO epochs | One |
| Explicit KL and entropy objectives | Disabled, as in the released 9B configuration; still record drift and entropy diagnostics |
| Learning rate | Constant 1e-6, no warmup |
| Weight decay and gradient clip | 0.01 and 1.0 |
| Sampling | Temperature 1.0, top-p 0.95, top-k 20 |
| Affine group size | Eight instead of the reference sixteen |
| Affine training target | Up to 512 distinct task groups in one optimizer step, instead of the reference 64 |

Sources: pinned [configuration](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/recipes/code/config/train.yaml),
[launch settings](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/scripts/code/train.sh)
and [loss implementation](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/verl/trainer/ppo/core_algos.py).

The reference v1 OLD path recomputes probabilities with the training engine
when `algorithm.rollout_correction.bypass_mode=false`. The separate
`use_rollout_log_probs` flag does not switch that path. Preserve this behavior
for the initial reference baseline. Record claimed miner sampling probabilities,
trusted recomputed sampling probabilities and training-engine OLD probabilities
under distinct names. Signing or hashing a miner's claimed numbers does not
verify their numerical correctness. Never use unaudited miner values as OLD.
Do not silently
substitute our private experiment's generation-probability denominator or mix
this recipe with the different frontier-model algorithm.

Freeze OLD against the epoch input before the update. Verify the executed path,
normalization and finite ratios with the actual sampled data. Inspect and record
the instantiated optimizer's betas, epsilon, precision and clipping; YAML values
alone do not prove what the Megatron adapter forwards. If using CPU offload
requires a different optimizer implementation, validate update and moment parity
against the pinned reference and identify that implementation explicitly.

The released prompt-mean trainer expects its complete collected batch to fit
one optimizer minibatch. Set both sizes to the same number of selected groups.
At 512 groups this is 4,096 trajectories. Accumulate microbatches to fit memory,
then update once. Do not split the same 512-task collection into eight successive
64-task updates while recomputing OLD from each changed actor. Use a smaller
complete task batch if throughput measurements require it; never trim individual
groups or silently change the loss normalization.

References for these execution details:
[trainer](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/verl/trainer/ppo/v1/trainer_base.py),
[optimizer adapter](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/verl/utils/megatron/optimizer.py).

## Sampler and termination qualification

The prepared MiMo sampler implements top-k/top-p composition, stable tie
handling, tokenizer vocabulary support and final renormalization in generation,
fast prefill verification and exact fallback. The signed contract binds these
settings. CPU boundary controls and private H200 checks are in progress; old
numerical calibration cannot be reused for the new model or distribution.
The model's padded output head is larger than its tokenizer support; sampling
and verification must exclude the padding IDs consistently.

Sampler probabilities after filtering and training-engine probabilities may
have different support. Keep these definitions explicit in artifacts and tests.
Log numeric disagreements and use the existing indeterminate/fallback path;
do not label an ambiguous support boundary as confirmed cheating.

Proposed length qualification compares 4,096 and 8,192 generated tokens, with
8,192 the launch target if measured completion, memory and latency permit it.
Set total context separately to accommodate the trusted prompt. The chosen
model's maximum advertised context is not the fleet's training capacity.

Publish a separate termination policy. The new manifest pins
`mimo-final-channel-math-v1`: for thinking-enabled generation, one leading
`<think>…</think>` channel precedes the final assistant answer. Grade the completed
boxed answer in that final channel; boxes inside the thinking channel do not
replace it. Paired LaTeX wrappers are accepted. Preserve the entire original
token sequence for sampling verification and training. Missing or malformed
finals, unclosed thinking and budget caps remain unresolved. This rule establishes
the final outcome, not the correctness of every reasoning step.

Correct completed responses have reward1;
completed wrong responses have reward0. Budget caps remain unresolved and cannot
satisfy the wrong-answer quota. The implemented Affine adaptation retains their
reward0 in the group mean but masks their entire trajectory out of policy loss.
Declare this explicitly rather than attributing it to MiMo's
`mask_unfinished_episode=false`. Allan's prefix mask and fixed-1024 normalization
would be separate later experiments.

## Parallel implementation and tests

1. Trainer track: pin upstream, implement the math batch adapter, match loss and
   gradients against reference fixtures, verify microbatch partition invariance,
   and qualify the actual optimizer and checkpoint publication lifecycle.
2. Protocol track: compose the candidate miner, manifest resolver, capture,
   admission, sampler and verifier with shared versioned rules. Cover all eight
   nonces, task/prompt binding, outcome swaps, repeated uploads, stale checkpoints,
   token mutations, manufactured answers and legitimate repeated outputs.
3. Model and evaluation track: compare starting models, pin tokenizer/template,
   establish the new baseline and run the real H200 sampler/backward checks.
   Audit task and grader visibility for answer leakage as well as model errors.
4. Operations track: stage source registries, public discovery, node runtimes,
   recovery and documentation. Preserve the hourly writer and its transaction
   cursor throughout. Do not reuse GPUs still owned by original experiments.

Before public activation, require the actual miner → public upload transport →
native math grader → GRPO update → model publication path to complete two private
epochs, plus an empty/no-update interlude and a lost-ack recovery test. Run honest
generation and verification across two H200s. CPU fixture success alone is not
GPU or production qualification. Check external launcher compatibility before
opening the public successor.

## Public cutover and rollback

Publish the reviewed GitHub code, dependency pins, resolved trainer settings and
immutable source archive first. Update `dashboard/public/llms.txt`, miner setup
instructions and discovery together. Supported supervisors can fetch a newly
approved child bundle; fixed-source or custom clients may need an upgrade.
Advertise required client features and fail clearly when unsupported.

Complete the old epoch with its original contract. Create a new run ID using
the chosen upstream weights and a fresh optimizer. Archive the previous run's
model, reports and lineage; do not carry its optimizer into the reset. Stage the
new verifier and trainer versions before atomically publishing the successor
manifest. Switch miners through the authenticated current-checkpoint handover.
Model publication must finish and verify before that handover becomes visible.

Use the existing dashboard appearance. Start a separate run series with its
own baseline and preserve old history. Publish finalized training/evaluation
evidence and the active protocol description without exposing open submissions.

Keep hourly weights on the best authenticated assessment mapped to current
UIDs. Retain prior valid evidence through the transition; model reset does not
reset earned assessments. Keep the sole writer, cursor and pending commitments.
Never substitute owner-only weights or restart the archived controller/burn units.

For a failed new run, reconcile the original update transaction before retrying.
An uncertain acknowledgement must not cause a duplicate training update. Return
the public pointer only to a complete compatible signed release, maintaining
distinct run histories and continuing hourly weights while training recovers.

## Fleet and operating targets

Keep one logical trainer, the existing eight verifiers, a separate evaluator,
test miner capacity, and coordinator/validator on Arbos.life. Benchmark the
reference backend on available resources before choosing its physical trainer
topology. Full 8–9B training needs optimizer offload or sharding; fitting inference
on an H200 does not establish training fit. Rent additional training capacity
only when the runnable configuration and measured memory needs justify it.

Proposed one-hour budget: collection 20 minutes, capture/preparation 5, training 25,
publication 7 and reserve 3. These are qualification targets, not measured promises
for the new model. Audits and held-out evaluation continue independently. Measure
all stages and use the largest complete task batch that fits the budget.

Keep model files for the working lifetime of their checkpoint. Reuse loaded
models across jobs; retire obsolete versions only after their leases finish.
Upload model checkpoints and evidence, retain optimizer state locally with
bounded recovery generations, and remove completed local rollout caches after
durable receipt. Exercise this in normal code paths rather than manual cleanup.

## Learning evidence and completion criteria

Establish a fresh held-out baseline with fixed prompts, sampler and token budget.
Track correctness, cap/EOS rates, token lengths, entropy, clipping/ratio diagnostics,
gradient norm, effective LR, group rewards, task coverage and train-data performance.
Record exact input/output checkpoints and the groups actually trained on.

After functional qualification, compare the fresh-run curve against its own
baseline at predetermined update counts, initially 16, 32 and 64, with a separate
confirmation evaluation for any improvement claim. Batch changes, learning-rate
changes and optimizer changes each require logged prospective settings. Preserve
the original private experiments and capture their original results.

Technical cutover is complete when compatible external miners submit full groups,
independent audits work, successive epochs publish actual model updates, hourly
weights continue correctly and public documentation matches the active release.
That milestone is separate from demonstrating sustained held-out improvement.
