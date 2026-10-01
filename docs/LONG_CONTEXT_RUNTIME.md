# Isolated long-context model computation

`subnet/long_context_runtime.py` is a separate runtime, not a selector used by
the deployed controllers. It pins Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, seven exact file hashes and a 32,768
token context limit. Weights are retained on the existing GPU pod; no weights
are downloaded to the operator machine.

The version `cuda-bf16-sdpa-flash-sm86-selective-head-v1` requires CUDA sm86,
BF16, explicit PyTorch SDPA FLASH_ATTENTION, deterministic algorithms, TF32
disabled, no cache and two TOPLOC threads. Unsupported flash attention fails;
there is no fallback to a different backend. Every token enters the base model.
Only the rows predicting output tokens enter the output head, producing full
vocabulary log probabilities. TOPLOC fingerprints all final-layer prefix and
output activations. This is a new numerical profile, not equivalence to the
existing eager runtime.

The operator-signed job binds source and helper hashes, exact model files,
interpreter/package versions, role and resource guards before model/artifact
reads. Each model load waits for at least 12 GiB free VRAM. Existing jobs are
preserved. Credentials and private signing keys are never sent to the worker.

## Measured initial control

`state/long-context-eog/probe-17485-1790842340` contains the fetched report,
artifact, probabilities and signed operator audit. The genuine greedy model run
used 17,485 prefix tokens and four output tokens, with probability shape
4 × 151,936. A freshly loaded independent model passed log-probability checks
at absolute tolerance 1e-5, relative tolerance zero, and zero TOPLOC errors.
Output-token, prefix-token, log-probability, malformed-proof and profile
mutations were rejected. Peak allocated GPU memory was 1,703,115,776 bytes and
the entire control took about 16 seconds. The process exited and released its
GPU allocation. This is a numerical runtime control, not a native EOG solve or
quality improvement.

The initial source bytes are preserved in
`state/long-context-eog/approved-source-v1`. The independently checked HF weight
metadata matches SHA256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.

## Stronger controls and native input binding

`ops/probe_long_context_adversarial.py` is a separate approved experiment. It
tests an altered but correctly framed polynomial proof and outputs,
probabilities and proofs computed after a real in-memory decoder-weight change.
The verifier freshly reloads the approved files. It also recomputes a complete,
self-consistent bundle on an edited early prefix, then checks that the signed
expected-input commitment rejects that bundle. Numerical consistency alone
does not prove the inputs are the authorized task inputs. Native tool responses
must additionally bind to authenticated environment replay.

Native EOG admission remains separate and requires the complete public
messages, tool schemas, actions and observations plus the original private
grader replay. Private seeds, database snapshots and grader queries are not
sent to the model worker. Curated tool-action computation does not establish
that the model originally sampled those actions.

## Larger-model budget

Metadata pins Qwen2.5-1.5B-Instruct revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, with 3,087,467,144 bytes of weights.
BF16 parameters plus gradients need roughly 6.17 GB; two Adam moments add
roughly 6.17 GB in BF16 or 12.35 GB in FP32, before activations, optimizer
workspace and other jobs. Full long-context training therefore needs an actual
idle-device measurement with gradient checkpointing and an explicitly pinned
optimizer/state dtype policy. This estimate is not a fit or training result.

## Native EOG paired computation and full optimizer control

The frozen six-tool controls are retained at
`state/long-context-eog/eog-positive-1790843483` and
`state/long-context-eog/eog-negative-1790843706`. Together they contain all twenty-four
JSON/probability files across the pair (twelve per control), signed model audits and exact approved
model profiles. Fresh model loads independently verified every turn. The
separate frozen native-v2 broker replayed the actual tool calls and original
private grader, producing rewards 1 and 0 in
`state/native-eog-split/model-admission-{positive,negative}-v2.json`.
These are curated target-model computations, not evidence that the model
originally sampled the actions. The old six-call controls have no generated
terminal assistant reply and are not common-epoch admissions.

Qwen-tokenized prompt lengths are 5,991 through 13,465 for the positive trace;
the separate numerical control above exercises 17,485 tokens. The pair retains
345,198,592 raw probability bytes and 165,472,827 bytes of honest JSON/compressed
arrays. It exceeds the deployed 100 MB upload budget. No probabilities were
quantized or truncated. A prospective isolated transport can explicitly allow
250 MB compressed while retaining the 500 MB raw guard; the deployed budget
has not changed. Exact worker bytes and both signed jobs are preserved under
`state/long-context-eog/approved-eog-source-v1`.

`subnet/long_context_training.py` provides a differentiable selective-head
sequence log probability with the same complete transformer context.
`ops/probe_long_context_optimizer.py` ran an actual one-step full-parameter
control in `state/long-context-eog/full-optimizer-1790844849`. The matched
fourth-turn prompt has 10,806 tokens, with 83 chosen and 73 rejected tokens.
The objective uses mean token log probabilities and reference-relative
preference loss (beta 0.1), AdamW learning rate 5e-5, no weight decay, clipping
at 1, BF16 parameters/gradients/Adam moments, no master parameter copy, gradient
checkpointing and no cache. Auxiliary/environment text contributes context,
not output loss. This shorter negative candidate is a frozen experimental
control; future candidate-sampling policy requires separate balanced-length
qualification.

All 290 parameter tensors received gradients. The update changed 252 tensors
and 225,588,364 elements among 494,032,768 parameters. Peak allocated GPU memory
was 4,601,926,656 bytes under the signed 8 GiB allocator cap; the process waited
for 20 GiB free VRAM without restarting other jobs. The training-pair preference
margin increased by 1.29797, while both individual log probabilities fell.
This is training-fit evidence, not task-quality improvement. The new remote
checkpoint is `426c3ddeaacc849d1792b5fb22d8c3fc37f4aefabfa2ef7627dd82e13864ccd7`;
a fresh model reload verified its new computation at unchanged tolerances.
Exact optimizer sources and signed job are retained in
`state/long-context-eog/approved-optimizer-source-v1`.

`ops/publish_long_context_checkpoint.py` delegates only six exact-object PUT
URLs to the worker after verifying the authorized completed training report.
The operator independently streams every published object to check its hash
and size before signing an immutable checkpoint descriptor. Account
credentials and signing keys stay on the operator machine. Publication does
not change a deployed checkpoint pointer or make the controlled run payable.

## Prospective common runtime and bounded gradient accumulation

`subnet/long_context_service_runtime.py` exposes
`LongContextServiceRuntime(checkpoint, files, environment, harness,
session_factory)` with `.configure`, `.for_environment`, `.rollout` and
`.verify`. A trusted deployment supplies `session_factory(spec)`; a miner
artifact never supplies executable factory code. Native EOG sessions use the
separate public-actor/private-grader adapter. This module is not selected by
the active controllers. Its distinct service revision additionally binds
sampling's NumPy float32 sequence-score reduction and the service source hash.
Generation and verification preserve the full 32K budget, all schemas and
messages, selective output heads, full probabilities and strict TOPLOC/native
replay. It cannot inherit the old runtime's 8192-token cap or full-prefix head.

`subnet/long_context_service_training.py` defines the separately versioned
`bf16-full-sequential-agent-turn-dpo-mean-v1` policy. It computes a scalar
reference-relative margin at unchanged weights, derives the preference-loss
coefficient, then backpropagates one token-weighted agent turn at a time before
one optimizer step. This avoids retaining fourteen long-context graphs for a
seven-turn pair. The mathematical chain-rule equivalence is tested against a
full graph with unequal turn lengths; BF16 accumulation order is explicitly a
new numerical training policy. Non-agent model roles cannot enter the loss.
The helper enforces an 8 GiB process allocator cap, BF16 Adam moments and full
finite gradients. The complete seven-turn optimizer is not yet measured by the
single-turn control above; admission requires its own actual bounded run.

The fresh terminal model controls at
`state/long-context-eog/eog-terminal-{positive,negative}-1790846259` use the
published trained checkpoint and independently recompute all six tool calls
plus a curated `DONE` response after the complete post-tool history. Neither
reuses old six-turn proofs. Both passed all seven strict model checks. A new
public Building-2 negative replaces the shorter negative only in this new
control; the historical negative remains frozen. The first divergent outputs
are both 83 tokens under a shared 10,806-token prompt. The resulting SUM-score
candidate policy at temperature 4 assigns probabilities 0.36351 and 0.63649
to positive and negative, respectively; this is not a quality improvement.
The pair retains 364,646,400 raw probability bytes and 173,094,112 honest
artifact bytes. Fresh original native terminal grading passed rewards 1 and 0 in
`state/native-eog-balanced-terminal-v3/model-admission-{positive,negative}-seven-turn.json`.
The signed combined receipt is
`state/long-context-eog/balanced-terminal-v3-completion.json`. The admission
binds the seventh model-computed DONE response, exact six native calls, final
scalar outcome and private original grader replay. These are controlled curated
model/native controls; they do not mark a common epoch verified.
