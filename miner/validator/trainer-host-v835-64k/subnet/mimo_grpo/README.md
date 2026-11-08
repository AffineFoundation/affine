# MiMo-reference GRPO training adapter

This package is a prospective trainer, not an activated network contract.
It invokes the actual pinned MiMo `compute_grpo_outcome_advantage`,
`compute_prompt_loss_weights`, `agg_loss` and `compute_policy_loss_vanilla`
function bodies. The complete upstream files, license and hashes are under
`vendor/mimo_verl_pinned`; include that directory in worker source bundles.
Only import and registry plumbing is adapted. No coding-agent stack is loaded.

The resolved baseline is constant learning rate 1e-6, AdamW weight decay .01,
betas (.9,.999), epsilon 1e-8, global gradient clipping 1, one PPO epoch,
ratio clip .2, negative-advantage dual clip 3, no explicit KL/entropy loss,
and no standard-deviation division of group advantages. Sampling contract is
T1/top-p .95/top-k20; these filters are not reapplied to training-engine OLD.

## Inputs and boundaries

The caller supplies authenticated run/epoch/checkpoint/contract identity,
native task membership, regenerated prompt tokens, all nonce attempts 0..N-1,
and native grades in `TaskGroup` objects. `admission_sha256` identifies that
trusted preparation record; its presence alone does not authenticate it.
The trainer is not an alternate miner admission endpoint. It accepts no
miner-supplied logprobs as PPO denominators.

`train_update(runtime, optimizer, groups, binding, config=..., guard=...,
record=..., microbatch_rollouts=1)` returns a complete update report.
The runtime supplies the actual model and qualified differentiable scorer:

```
runtime.model
runtime.checkpoint_id
runtime.vocab_size
runtime.eos_token_ids
runtime.max_context_tokens
runtime.score(prompt_ids, output_ids, gradients=bool) -> Tensor[length]
runtime.save_model(directory)
```

The scorer returns raw temperature-1 selected-token training log probabilities.
Compute OLD for the whole selected optimizer batch before any update, matching
upstream v1 `algorithm.rollout_correction.bypass_mode=false`. The separate
`use_rollout_log_probs` YAML flag does not switch this executed upstream path.
Record sampler LP claims elsewhere under a distinct name. A signed claim is
not trusted numerical evidence.

The model remains in eval mode (dropout disabled) with gradients enabled for
backward. Sampling is external to this service. OLD/current kernel differences
are bounded by an explicit qualified tolerance; an excess blocks the training
update and is not a cheating verdict. Global prompt weights are computed before
microbatch partitioning. Up to 512 distinct complete task groups accumulate into
exactly one optimizer step. There is no successive 64-task-update substitution.

## Explicit Affine adaptations

- Group size defaults to eight rather than MiMo's sixteen.
- Up to 512 tasks rather than its released 64-task launch default.
- Unresolved trajectories remain reward zero in the all-N group statistics,
  but their entire policy-token loss is masked by default. This includes both
  budget caps and EOS responses with no gradeable final answer. They never
  satisfy the completed-wrong admission minimum. The alternative named
  unresolved-active policy is for prospective qualification only; neither
  interpretation is claimed to reproduce an undocumented upstream grader.
- Actual PyTorch `AdamW(foreach=False, fused=False)` updates FP32 CPU master
  weights and moments, with parameter-at-a-time gradient transfer. GPU BF16
  weights are projections. This replaces upstream's distributed Megatron
  optimizer transport; it is not represented as running Megatron. CPU fixtures
  compare all masters/moments/counters bit-for-bit to one ordinary PyTorch
  AdamW update. Actual GPU/adapter numerical qualification remains required.
- FP32 gradient buffers accumulate weighted microbatch gradients before one
  global norm clip. Increasing effective batch size does not require all
  trajectory graphs to remain resident at once.

## Lifecycle integration

Use `execute_transaction` with the coordinator's authenticated original job,
sole trainer lease callback and immutable model publisher. It reuses the
complete-group candidate's updating/staged/published journal, with provenance
recorded in `TRANSACTION_PROVENANCE.json`. Its `stage_for_publication` writes
actual model safetensors plus assets and bounded local optimizer shards.
The returned `publish_files` and `local_state_files` are disjoint. Only the
model inventory goes to R2. `CPUAdamW.restore_local` requires the authenticated
descriptor hash and verifies all local chunk hashes and checkpoint projection.

The publisher must verify full model readback before returning its original
receipt. The host publishes discovery only after that transaction succeeds.
An uncertain upload acknowledgement resumes publication, without another
optimizer step. An interrupted update remains blocked for explicit recovery.
`restore_transaction_state` authenticates the original Job and staged files,
reloads the exported model and restores FP32 optimizer state in a cold process.
The stage operation refuses to overwrite an existing generation. Empty updates
do not advance Adam. Authenticated publication handover updates the persistent
runtime and optimizer checkpoint identity without a model reload.

Capacity must be admitted before constructing a full-model optimizer. Persistent
CPU master/moments require 12 bytes/trainable parameter plus bounded scratch;
GPU FP32 accumulation requires four bytes/trainable parameter. Local generations
require about 12 bytes/parameter each. `admit_fresh_optimizer` checks actual
cgroup-aware RAM and free disk before a full-model qualification starts. The
transaction retains one current committed optimizer generation plus one new
in-flight generation, and automatically retires only obsolete published
optimizer directories after replacement is durable. Model cache retention is
owned by the runtime; model history and reports remain in the durable bucket.

The detailed recipe, private worker command and boundaries are in
`docs/MIMO_GRPO_IMPLEMENTATION.md`. The qualification command consumes the
coordinator's signed opening/dispatch plus the exact committed compact documents;
it does not invent labels or claim a sampling audit happened.

Run the CPU suite with pytest and the repository torch/safetensors environment:
`python -m pytest tests/test_mimo_grpo.py`. Passing these tests does not qualify
the new model's H200 kernels, cross-machine inference proof or live publication.
