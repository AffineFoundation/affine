# MiMo GRPO trainer implementation

This is an implemented prospective training path, with CPU integration tests.
It is not a statement that the public network has switched, that the MiMo H200
runtime is qualified, or that held-out performance has improved. The public
`llms.txt` must change with the eventual authenticated release, not merely
because these files exist.

## Exact reference and adaptations

The initial actor is `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`, loaded with the
actual `Qwen3_5ForConditionalGeneration` architecture, tokenizer and chat
template. Pin a specific model revision and complete file hashes through
`ops/mimo_fetch_checkpoint.py`; do not load an unpinned moving model name.
The hybrid model uses BF16 parameters and FP32 recurrent state. The qualified
profile records thinking mode, context, attention implementation and the complete
EOS set, initially `[248046, 248044]`. Text math training freezes the vision
tower and unused optional heads; the trainable parameter inventory is explicit.

`subnet/mimo_grpo/upstream.py` verifies the SHA256 of the vendored source and
executes unchanged function bodies from XiaomiMiMo/verl commit
`a2ad9f6160b03ff2d47e59832bfb6b289f37c917`:

- `compute_grpo_outcome_advantage`: reward minus group mean, with standard
  deviation normalization disabled.
- `compute_prompt_loss_weights` and `agg_loss`: active token mean within each
  task group, then mean across distinct task groups.
- `compute_policy_loss_vanilla`: PPO ratio clipping plus the negative-advantage
  dual clip.

Only imports, registry decorators and configuration plumbing are adapted.
Full original files, Apache license, source URLs and hashes are retained in
`vendor/mimo_verl_pinned/PROVENANCE.json`. This is not the old preference loss
renamed as GRPO, and it does not import the upstream coding-agent environment.
See the pinned [loss functions](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/verl/trainer/ppo/core_algos.py),
[v1 trainer](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/verl/trainer/ppo/v1/trainer_base.py),
[configuration](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/recipes/code/config/train.yaml)
and [launch script](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/scripts/code/train.sh).

| Parameter | Executed initial value |
| --- | --- |
| Optimizer | PyTorch AdamW, CPU FP32 master weights and moments |
| Learning rate | Constant `1e-6`, no warmup |
| Weight decay | `0.01` |
| Adam betas, epsilon | `(0.9, 0.999)`, `1e-8` |
| Gradient accumulation | FP32; one global norm clip at `1.0` |
| PPO epochs | `1` |
| PPO clipping | Ratio `[0.8, 1.2]`; negative-advantage dual clip `3.0` |
| KL coefficient, entropy coefficient | `0`, `0` |
| Advantage standard-deviation normalization | Off |
| Sampling | Temperature `1.0`, top-p `0.95`, top-k `20` |
| Complete trajectories per task | Initially `8`, nonces `0..7` |
| Eligible group | At least one completed correct and one completed wrong |
| Optimizer batch | Up to `512` distinct eligible task groups |
| Execution microbatch | One rollout graph by default; gradients accumulate |

Eight trajectories instead of the reference sixteen and up to 512 tasks instead
of 64 are deliberate Affine adaptations. At capacity that is 4,096 trajectories
in one optimizer update. The trainer does not silently turn them into eight
successive 64-task updates. Global task weights are calculated before partitioning
into microbatches, so partition size does not alter the intended objective.

The CPU FP32 optimizer uses actual `torch.optim.AdamW(foreach=False, fused=False)`.
Parameter-at-a-time gradient transfers bound host scratch memory. CPU fixtures
compare master weights, both moments and every counter bit-for-bit to ordinary
PyTorch AdamW over multiple steps. This is an explicitly named replacement for
upstream Megatron optimizer transport, not a claim to execute Megatron or have
already established cross-runtime GPU update parity.

The prospective default decay policy is `mcore-bias-and-1d-excluded-v1`: apply
`0.01` to ordinary weights, including embeddings, and zero to names ending in
`.bias` or tensors with one dimension. This matches the default grouping in
[Megatron Core 0.15.0](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.15.0/megatron/core/optimizer/__init__.py),
which the pinned MiMo wrapper delegates to without a custom exemption. The
resolved policy and parameter-group inventory digest are reported and bound to
local optimizer state. The frozen private v4/v6 rehearsal applied decay to all
trainable parameters; that declared adaptation remains preserved in its source
and reports. New code offers it only as explicit
`affine-all-trainable-decay-v1`, and does not silently reinterpret an old state.

## OLD probabilities and unresolved outcomes

Before any gradient or parameter update, the trusted trainer prefills every
active trajectory against the immutable input checkpoint and records selected
token log probabilities. These are raw temperature-1 model log probabilities.
Top-k/top-p filters define generation but are not applied again to training OLD.
Miner-supplied probabilities are never PPO denominators.

This follows the executed upstream v1 path with
`algorithm.rollout_correction.bypass_mode=false`. The separate YAML
`use_rollout_log_probs` flag does not switch that path. The input model stays
unchanged until all groups have accumulated; the implementation checks parameter
versions and OLD/current forward consistency. The initial same-training-engine
logprob drift limit is `1e-5`, which requires actual H200 qualification before a
live release. Exceeding it is a numerical/runtime failure, not miner fraud.

Native grading is separate from sampling audits. Budget caps and EOS responses
without a gradeable final answer are unresolved. Their rewards remain zero in
the mean of all N original attempts, but all of their policy-loss tokens are
masked under `affine-unresolved-zero-reward-mask-all-v1`. They cannot satisfy
the completed-wrong minimum. This conservative masking is a declared Affine
adaptation; it is not attributed to an undocumented upstream grader. Original
unresolved trajectories stay in the compact training documents and reports.
A grader/infrastructure error defers a group rather than assigning reward zero.

Different assigned nonces may legitimately produce identical output tokens.
The complete-group contract keeps those outcomes. It rejects missing/repeated
nonces and conflicting attempts instead of removing naturally repeated text.
Mixed-group admission and miner task search still condition the task distribution;
complete groups do not prove unbiased task selection or historical execution order.

## Worker API and trust boundary

`complete_group_native.grade_batch` regenerates task prompts, decodes exact
submitted tokens, runs the native grader and returns every authoritative native
result. `mimo_coordinator.Coordinator.seal` signs the selection and those results.
No model sampling audit is required before an eligible group can reach training.

`prepare_signed_dispatch(opening, dispatch, authority=..., documents_root=...,
runtime=..., optimizer_step=...)` verifies host signatures, manifest identity,
checkpoint/runtime pins and every compact document hash. Local documents are
named `<sha256>.json`. It invokes `project_document`, producing `TaskGroup`s
whose native admission receipts bind the exact task, prompt, all output tokens,
grades, nonce set and checkpoint. Never pass miner-supplied outcomes into this
trusted API as if they were native results.

`train_update(runtime, optimizer, groups, binding, config=..., guard=...,
record=..., microbatch_rollouts=1)` performs one optimizer update. Its report
records input checkpoint, group/admission hashes, outcome counts, active and
masked tokens, per-trajectory loss, trusted OLD/current probability diagnostics,
unclipped gradient norm, actual optimizer settings/counters and stage timings.
An empty eligible set produces `no-update` without advancing the optimizer.

`execute_transaction` wraps that update with the original candidate's leased
transaction journal. Provenance is in
`subnet/mimo_grpo/TRANSACTION_PROVENANCE.json`. The host provides:

1. Authenticated original manifest/source hashes and `InputBinding`.
2. A sole-trainer lease callback, maintained across all epochs and recovery.
3. An immutable, idempotent model publication callback with full readback.

The callback receives only model safetensors/config/tokenizer paths. Optimizer
masters, moments and counters stay local. After verified model publication,
the persistent runtime and optimizer move to the new checkpoint identity without
reloading the model. Public discovery moves only when the coordinator separately
commits this result. The trainer does not change chain weights.

After a process restart, `restore_transaction_state(workspace,
expected_job=original_authenticated_job, load_runtime=..., config=...,
assert_trainer_lease=...)` verifies the original journal, report, model and all
optimizer shards, loads the actual output model, and restores FP32 state. A
subsequent `execute_transaction` with the original inputs completes or returns
the same update, without another optimizer step. A partial `updating` state
cannot be retried automatically; it requires explicit original-state recovery.

## Capacity and local lifetime

CPU persistent state costs roughly 12 bytes per trainable parameter, plus
bounded per-parameter Adam scratch. GPU FP32 accumulated gradients cost another
4 bytes per trainable parameter in addition to the model and backward graphs.
Use `admit_fresh_optimizer` before allocating the full optimizer; it checks
actual cgroup-aware available RAM and free disk. It does not substitute for an
actual long-context forward/backward capacity test.

The transaction retains one current committed optimizer generation and at most
one in-flight replacement. After durable replacement it automatically retires
obsolete published optimizer directories, preserving journals/reports. It never
removes the last committed generation before publication succeeds. Model cache
leases and bounded retention are handled by `mimo_runtime.CheckpointCache`;
the active checkpoint is not deleted after each audit job. Durable model,
rollout and report history remains in the bucket.

## Private real-model qualification

Use a separate nonpayable namespace and the dependency/runtime pin prepared on
the target H200. Include `subnet/mimo_grpo/` and `vendor/mimo_verl_pinned/` in
the authenticated source bundle. Fetch the pinned model on the GPU worker,
then prepare its runtime profile with `ops/mimo_qualification.py prepare`.
The opening manifest must contain this exact profile and checkpoint descriptor.

Run the real miner/native grading path first and stage the signed opening,
signed dispatch and exact captured compact documents. No synthetic positive
or negative labels are generated by the training command. If the miner produces
no eligible complete group, report that outcome instead of fabricating a pass.

```bash
python ops/mimo_training_qualification.py \
  --checkpoint /worker/checkpoints/INPUT_ID \
  --pin /worker/input/runtime-pin.json \
  --pin-sha256 OPERATOR_AUTHENTICATED_CANONICAL_PIN_SHA256 \
  --opening /worker/input/OPENING.json \
  --dispatch /worker/input/DISPATCH.json \
  --authority COORDINATOR_PUBLIC_ED25519_HEX \
  --documents-root /worker/input/documents \
  --source-closure-sha256 AUTHENTICATED_SOURCE_BUNDLE_SHA256 \
  --workspace /worker/private-training-check-1 \
  --output /worker/private-training-check-1/report.json \
  --max-groups 1
```

The caller authenticates the source bundle and expected pin hash before launch.
The command requires a new workspace and verifies the pin hash, signatures and
input hashes itself. It does one actual GRPO update on full original trajectories,
exports only the model to a local readback directory, then destroys and reloads
the runtime/optimizer from the exported state. It compares a streaming digest
of all masters/moments/counters and output probabilities before and after restore.
It records how many groups were omitted for a short functional diagnostic.
No trajectory is shortened or relabeled for convenience.

This command's receipt is explicitly local diagnostic readback, not R2
qualification. It never publishes discovery or sets weights, and its report
always says `qualification=false`, `sampling_qualification=false` and
`convergence_evidence=false`. Follow it with the coordinator's real R2 lifecycle,
two consecutive private epochs, cross-machine sampling controls, and fixed-heldout
evaluation before deciding to activate a public release.

## Signed queued trainer worker

The existing worker entry point routes `role=train` jobs with
`worker_policy=mimo-complete-group-worker-v1` to `mimo_training_worker`.
It checks signed opening/job envelopes, the exact installed source and runtime,
and the assigned `trainer_worker` before fetching training artifacts. The job's
`training` object contains `run_id`, `optimizer_step`, the exact `config`
document, a signed `dispatch` URL/hash/size, `documents` indexed by their content
hashes, `publication_namespace`, `publisher_id`, and `role_config_file`.

The role configuration stays in an owner-only local file. It supplies the
existing bucket configuration, the publisher identity file and an optional
already-downloaded initial checkpoint. Reports never contain credentials.
Compact signed native receipts carry one trusted prompt per group and output
hashes; the trainer reconstructs output tokens only from the original captured
documents after verifying those hashes. This is native grading admission, not
an expensive sampling-verification barrier.

`R2ModelPublisher` streams only model safetensors/config/tokenizer artifacts to
immutable checkpoint-addressed keys. Conditional creation prevents replacement;
every artifact is subsequently downloaded and SHA-256 checked. Its signed
receipt binds the original job, source closure, input/output checkpoints,
optimizer step, report hash and exact model inventory. The receipt excludes
optimizer state. Retrying a lost acknowledgement returns the same publication
without a second optimizer update.

The worker commits its local `CURRENT` record after verified publication and
before retiring the previous model and optimizer generation. It retains the
current generation and one in-flight replacement, plus small journals/reports.
Admission checks free disk before a fresh update stages that replacement.
This lifetime spans a checkpoint's work; acknowledgement does not delete the
active model after each batch. A failed or partly restored optimizer is poisoned
and cannot silently resume with partial state.

Queue execution requires a live ownership/lease callback and an explicit stable
`--trainer-state-root` outside the per-attempt job workspace. Ownership is checked
before training, the Adam step, staging, publication and `CURRENT` advancement.
Retries therefore recover the original transaction even when their lease-attempt
directories differ. Trainer jobs are assigned to one named worker; moving the
trainer to another machine requires explicitly transferring or recovering its
local optimizer state. An empty eligible set produces an authenticated no-update
report and does not advance the checkpoint or optimizer.

CPU regression command: `python -m pytest tests/test_mimo_grpo.py tests/test_mimo_grpo_publication.py tests/test_mimo_grpo_worker.py tests/test_mimo_worker.py`.
Runtime controls: `python -m pytest tests/test_mimo_runtime.py`.
These cover reference loss/gradient parity, accumulation partition invariance,
native unresolved masking, persistent optimizer parity, local state restore,
two updates, lost acknowledgements and cold-process publication recovery.

## Explicit task assets in releases

Native task snapshots must travel with the public source release. The approved
MATH snapshot is `assets/original-math7496.tasks.json`, 8,146,242 bytes, SHA256
`77a4524abc279d0e6e95ec87d0e5604f501c8e409ac5ab656ecabf060ab50fe3`.
Declare it in release settings before running `ops/build_mimo_release.py`:

```json
"task_assets": {
  "assets/original-math7496.tasks.json": {
    "sha256": "77a4524abc279d0e6e95ec87d0e5604f501c8e409ac5ab656ecabf060ab50fe3",
    "bytes": 8146242
  }
}
```

`mimo_release.build` includes exactly these declared assets, checks their bytes
against the inventory, and binds them into the source/archive hashes. It does
not scan `assets/` automatically. Files are limited to 16 MiB each; undeclared
assets, path escapes, symlinks and mismatched pins fail preparation. Verification
requires the archive, source inventory and declared task inventory to agree.
Extracting the authenticated archive therefore provides the snapshot at the
path already named by the environment, without a separate manual copy. The
native adapter still checks its signed environment source hash before use.

The MATH dataset contains public task statements and reference answers; these
are required by its native grader. The mining/evaluation index split remains a
separate signed admission rule: shipping the dataset does not authorize mining
held-out indices. Task assets contain no miner submissions or private credentials.
Historical frozen releases remain unchanged; they retain their original hashes.

### Trainer local storage admission

Run `python ops/prepare_mimo_worker_storage.py` during fresh worker bootstrap.
Its default root is `/workspace/affine-mimo-trainer`, on a local block-backed
filesystem. The helper rejects FUSE, network filesystems and tmpfs and checks
capacity for the committed optimizer/model generation, one in-flight generation,
a checkpoint cache and a reserve before creating owner-only role directories.
It does not delete or relocate any existing model, optimizer or experiment.

For the 9B text trainer, one local FP32 Adam state is approximately 107.45 GB and
one model export is 18.84 GB. Two generations plus the configured cache/reserve
require approximately 280.2 GB free at fresh bootstrap. Per-update admission
still measures actual parameters and free capacity. Keep `--trainer-state-root`
stable across job attempts; only model exports are published to R2. The original
MiMo rehearsal exposed very slow encrypted FUSE writes, so this path selection
is an operational requirement, not a reason to remove recoverable optimizer
state or repeatedly delete active checkpoints.

### Prospective qualified public release

Public trainer execution is disabled by default. The generic worker must pass its local `allow_public_mimo` setting through `mimo_worker` as `allow_public=True`; no submitted job field enables this. The manifest must also carry an enabled, authority-signed `mimo_release_policy` with all required qualification receipts. Training validates that policy again and requires its exact publication namespace (`mimo/<run_id>` for public releases, `private/mimo/<run_id>` for policy-bound private rehearsals).

Local trainer `CURRENT.json` records the policy hash and execution mode. Initialization and checkpoint handover bind the original epoch, input checkpoint, and run to that authenticated policy, and check the full output model inventory stays within its namespace. A successor manifest requires the authority's checkpoint authorization. Legacy isolated rehearsals remain supported without a release policy only for explicitly nonpayable private/test epochs. These hooks are prepared code; their existence does not activate public discovery or authorize deployment of an unqualified checkpoint.


### Model publication throughput

Model publication uploads and fully reads back up to four independent checkpoint files concurrently (`R2ModelPublisher.shard_concurrency`, range 1–8). Each file still uses bounded multipart buffers, conditional immutable writes, and a complete GET SHA256 check. Descriptor and signed receipt publication waits for every file to pass. The coordinator independently streams up to four files concurrently (`R2Services.readback_concurrency`, range 1–8), writing its small readback record only after all exact hashes match. Optimizer state remains local; no hash, lease or receipt requirement is removed. Historical sequential qualification runs retain their original source and timings.


### Parallel proof memory lifetime

Independent B1 replicas share a semaphore around full proof prefill. Each replica releases inactive CUDA allocator buffers before and after that phase, while still holding the semaphore. The model parameters remain resident. This is needed because serializing compute alone does not release another process's reserved proof buffers; four persistent allocator caches can otherwise exhaust an H200 on 8192-token inputs. The original 8192 qualification exposed this OOM and remains recorded as a failure. Longer-sequence memory and exact-output qualification is required before enabling the new profile; the 4096 exact replay alone does not qualify 8192 capacity.

Miner shutdown releases borrowed task views before their owning backend, clears strong model aliases, and then closes journals. A persistent verifier factory remains the owner of its borrowed backend. Replica cancellation terminates only owned children and preserves completed original nonce bytes; a native task error leaves that task terminal but allows a new process pool and proof semaphore for the next task. It never substitutes another nonce into the failed group. These lifecycle controls are tested across repeated epoch changes without depending on garbage collection.

The additional H200 memory control retained four maximum-length KV caches and five model copies while checking original 4467-token proof parity and synthetic 8192-token tensor geometry. That control passed; it is not evidence of a newly sampled, complete 8192-token group. Full opening-first mining qualification remains a separate acceptance check.
