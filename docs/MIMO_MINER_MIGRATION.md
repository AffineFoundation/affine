# Prepared MiMo miner migration

This describes the prospective release being privately qualified. It is **not
the currently active mining contract**. Keep following the authenticated public
manifest until the operator publishes the new run and its exact source/runtime
release. No public cutover time has been announced.

## What changes

The starting checkpoint will be the pinned
`XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` model, revision
`2367e865d009c13ac81713a2878291d33ab28177`. This checkpoint uses the Qwen3.5
hybrid attention architecture and MiMo's tokenizer/chat template. The approved
math dataset and separation between mining and held-out tasks remain in place.

One task group contains **all eight prescribed attempts, nonces 0 through 7**.
Submit them in that order. A group needs at least one completed correct answer
and one completed wrong answer. There is no four-correct/four-wrong quota.
Retain the original other outcomes, including unresolved attempts, in the group.
Do not replace attempts, change their nonces, add a seed offset, or search a new
nonce window for the same task. Parallel execution of the eight attempts is
allowed; the list order is a protocol rule, not proof of historical execution
order. Independently valid attempts may legitimately produce identical text.

Groups with all correct answers, all wrong answers, or no completed wrong answer
do not qualify. Search another approved task. An output that hits the token cap
without a completed answer is unresolved, not a failure. A grader/service error
also cannot satisfy the failure requirement. The initial limit remains nine
eligible task groups per registered UID per epoch. Re-uploading a group is an
idempotent retry and never earns another contribution.

## Runtime and verification

Use the exact signed checkpoint files, runtime fingerprint, source release,
thinking mode, context/output limits and sampler from the epoch manifest. The
prospective reference sampling settings are temperature1, top-p0.95 and top-k20.
Generation supports both pinned EOS tokens. Random draws bind the run, epoch,
checkpoint, miner identity, task, prompt, nonce and token position. Renewing a
checkpoint download URL does not change those draws.

The final-answer policy is `mimo-final-channel-math-v1`. For thinking-enabled
generation, a single `<think>…</think>` channel precedes the final answer. The
grader reads the completed boxed answer from the final channel. The entire
original reasoning and final token sequence remains in the proof and training
input. A correct final answer does not certify every reasoning step.

Proof uploads contain compact tokens, selected-token probabilities and TOPLOC
fingerprints, without full-vocabulary arrays. Verifiers recompute the model's
distributions. A prefill sampling disagreement is adjudicated with the prescribed
cached sampler before being classified as invalid. Native answer grading remains
separate from token-path verification.

The new transport sends immutable group/proof objects and compact training
documents using encrypted, per-miner R2 upload capabilities. It then updates a
small signed commitment. Preserve local upload state for retries; do not rewrite
an acknowledged immutable group. No personal bucket credentials are required.

## Updating a miner

The canonical miner resolves the new contract through the signed manifest and
uses a complete-group journal to resume the next missing attempt after an
interruption. Its MiMo runtime requires the qualified dependency lock and TOPLOC
wheel rebuilt against that exact torch build. Merely changing the model name or
installing an arbitrary TOPLOC wheel is insufficient.

An authenticated launcher capable of fetching a new approved source/runtime
release can perform the handover. Fixed-source or custom clients need an update
before mining the new contract. Unsupported feature sets fail explicitly; old
4/4 submissions are not silently reinterpreted as complete nonce groups.

The final announcement will provide the reviewed GitHub revision, immutable
source/runtime artifacts, manifest URL and cutover boundary. This preparation
document deliberately does not invent working public endpoints or claim that
the private diagnostic is a paid network epoch.

## Prepared launch procedure

The [prepared runtime inventory](https://affine.io/releases/mimo-runtime-20261013/ARTIFACTS.json)
now provides permanent public downloads for the exact dependency lock and rebuilt
TOPLOC wheel. Follow the [download, SHA256 verification and installation steps](MIMO_LIFECYCLE.md#reproducible-worker-installation)
to install a new runtime with `ops/bootstrap_mimo_runtime.py`. These artifacts are
marked prepared; wait for the authenticated MiMo opening before starting mining.
Use that runtime's Python to start the updated supervisor. On a qualified H200
with enough free memory, four independent
B1 replicas can generate the fixed attempts in parallel:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 /absolute/mimo-runtime/venv/bin/python -B -m subnet.miner_supervisor \
  --discovery-url https://affine.io/mining.json \
  --authority 3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd \
  --source-cache /absolute/private/mimo-approved-source \
  --key /absolute/private/miner.seed \
  --state /absolute/private/mimo-miner-state \
  --env-id affine_math --mimo-replicas 4
```

This is a prepared procedure for the future signed MiMo opening, not an instruction
to run the new runtime against an old Qwen epoch. The final activation notice and
current `llms.txt` will supply the actual release pins and qualification scope.
The local replica setting does not change N, nonces, seeds, quota, or submission
order. Default is one replica; two are also supported. All replicas must fit the
qualified GPU memory budget. A fresh model download can consume much of the first
collection window; downloads retain partial files and verify every complete hash.
MiMo checkpoint files download with up to four independent transfers. Completed
new checkpoint admission retires only obsolete checkpoints owned by this launcher.
Do not point a new runtime at another active launcher's state directory.

## Learning and rewards

The trainer uses the pinned MiMo reference GRPO loss with complete task groups,
trusted recomputed OLD probabilities, and gradient accumulation across distinct
tasks. Unresolved attempts have reward0 in the group statistic and their policy
loss is masked, an explicit Affine adaptation. Training can use natively graded
eligible groups while sampling audits run independently.

Contribution accounting remains per eligible task group, subject to the active
collision, audit and penalty policy. More correct trajectories inside a group do
not independently multiply its credit. Hourly weights continue from the best
authenticated miner assessments during the model transition.
