# Private fixed-group GRPO experiment

This package does not modify production. It runs a full-model, own-policy online
comparison from the exact original `2173b3e…` 7B checkpoint using the previously
qualified CPU FP32-master AdamW and GPU FP32-gradient accumulation implementation.
Only model weights are exported; original rollouts, selected-token logprobs,
grades, masks, update details and failures are retained for durable capture.

## Scientific contract

Each task receives all prescribed nonces `0..group_size-1` (initially eight).
Every result is retained before grading, including identical token paths from
different attempts. A task group qualifies only with at least one completed
correct and one completed incorrect answer. There is no 50/50 quota, response
selection or replacement nonce. Rejected whole groups are retained too.

An EOS answer lacking a final box is unresolved. A budget-capped answer is
always unresolved, including when a box appeared earlier. Unresolved rows remain
reward zero in the original eight-member mean/std calculation but contribute no
token loss, including KL. They never satisfy the completed-incorrect quota. This
is an explicit truncation-masked, mixed-group GRPO recipe, not an assertion that
it exactly reproduces every original-paper default.

Before generation, the original native grader checks each trusted reference
against itself. False or unparseable references consume a frozen candidate task
slot and are logged as whole-task skips. Runtime errors and timeouts abort the
original experiment; no model output is relabeled to hide an infrastructure error.

Candidate objective:

* Rewards 1 for correct, 0 otherwise; `(r - mean8)/(population_std8 + 1e-4)`.
* Per-token ratio `exp(new_selected_logprob - captured_behavior_logprob)`.
* PPO/GRPO clipped surrogate with epsilon 0.2.
* Differentiable sampled k3 KL `exp(ref-new) - (ref-new) - 1`, beta 0.01,
  frozen run-start reference. This is a sampled fixed-data KL surrogate, not an
  exact full-vocabulary KL or a detached reward-only KL implementation.
* Mean tokens per resolved trajectory, then divide by all eight original slots
  and actual admitted task count. Masked rows retain their denominator slots.
* One accumulated optimizer update per fresh collection. Clipping mainly guards
  generation/scoring drift because no inner policy epochs are used initially.

Control objective: mean of all correct × completed-incorrect logistic preference
pairs in each complete group, mean over tasks, beta 0.1, own input-checkpoint
reference reset each update. A scalar prepass computes exact coefficients;
streamed per-trajectory backwards reproduce the full pairwise first derivative
without retaining multiple long model graphs. This is an online group-preference
control, not a claim to reproduce production's balanced-quota data selection.

Both arms use fresh AdamW, LR 1e-6, betas .9/.999, eps 1e-8, weight decay .01,
clip norm 1, BF16 forward and FP32 master/moments/accumulation. They sample their
own current policies after each update. Task order is matched. The seed binds
the sampling-study identity, parameter checkpoint digest, optimizer step, pinned
runtime and sampler, task, nonce and token position. Later draws can therefore
differ across arms as their checkpoint changes; this is not a same-data contrast.

## Runtime and launch interface

From an operator-created workspace containing this package, `base/` and the
approved optimizer/native grader `source/` tree:

```
python -m research.fixed_group_grpo.worker --scope ROOT-SIGNED.json
```

ROOT owns signing, rentals, numerical canary admission, resource bounds,
baseline/endpoint evaluation, terminal supervision and durable capture.
The worker requires exact source/input/checkpoint hashes and current signed scope.
No signing key or storage credential is needed by the numerical worker.

`PLAN.json` fields: `policy`, `objective`, `steps`, `group_size`, `tasks_per_step`,
`min_tasks_per_step`, `max_candidate_tasks_per_step`, `learning_rate`,
`duration_seconds`, `disk_reserve_bytes`, `initial_ratio_max_token_logratio`,
`sampling_run_id`, `extend_context_without_rope_scaling`.
`INPUTS.json` contains `tasks`, a fixed flat schedule of
`steps * max_candidate_tasks_per_step` native task rows. Each update owns its
preassigned block, stops at the desired mixed-group count, or trains an allowed
underfilled group count at the block limit. Falling below the minimum stops the
original as incomplete. No threshold/seed/task adjustment occurs during a run.

Initial generation-versus-prefill token logratio mean/p99/max is written before
every admission check. ROOT's current proposed maximum is `log(1.2)`, which keeps
initial ratios in `[1/1.2, 1.2]`; this must be qualified on H200. The captured
generation denominator is never replaced with current-policy scores to conceal
numerical drift. All per-token old/current/reference LPs are small selected-token
vectors, not full-vocabulary artifacts.

The proposed 4096-output/8192-context profile exceeds upstream's nominal 4096
context. An explicit plan flag extends the model configuration limit without
changing RoPE parameters. This is a research runtime change applied identically
to both arms and their fresh baseline/evaluation; old-profile baseline scores
cannot be reused.

## Expected resources and limits

One H200 per arm, approximately 249 GiB host RAM: two BF16 model copies about
30.5 GB total on GPU; FP32 accumulation about 30.5 GB; single-trajectory activation,
logit and gradient working space in addition. CPU FP32 master plus two Adam
moments are about 91.5 GB, plus bounded per-parameter scratch. Actual capacity
admission remains mandatory. Do not retain optimizer snapshots per update.

At 16 updates × 32 mixed groups × 8 attempts, at least 4096 responses per arm
are admitted. A 96-candidate-per-update search ceiling permits at most 12288
generated responses per arm, up to 50,331,648 output tokens at a 4096 horizon.
Wall time depends strongly on capped outputs and group yield; no hours estimate
is claimed without H200 measurement. The provider TTL and ROOT execution scope
must reserve time for endpoint export and complete durable capture.

## CPU verification

```
.venv/bin/python -B -m unittest \
  research.fixed_group_grpo.test_objectives \
  research.fixed_group_grpo.test_training -v
```

These controls cover actual autograd signs, clipping, differentiable KL, matching
temperature/support, EOS and prompt masks, nonce coverage, legitimate repeated
text, whole-group outcome filtering, full factorized preference gradients,
real micro-backwards through the original FP32 accumulator, unresolved masks,
and fail-before-optimizer drift reporting. They do not replace remote numerical
qualification or demonstrate improved model accuracy.
