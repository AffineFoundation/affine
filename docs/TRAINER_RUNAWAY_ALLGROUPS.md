# Trainer change: train runaways, derail mask, constant normalization, all groups, k3 KL

Branch `fix/trainer-runaway-allgroups`. The branch is prepared only. Nothing is deployed.

## Behaviour
- `cap_policy = affine-unresolved-zero-reward-active-v1`. Unresolved rollouts (budget-capped,
  or EOS without a gradable answer) keep reward 0 in the group mean, and they are now TRAINED.
- Group admission: the new outcome policy `completed-eos-math-unresolved-failure-v2` counts
  `negative + neutral` toward `min_completed_failure`, so 4 correct + 4 runaway is admitted.
  Zero-variance groups are still rejected: all-correct fails `min_completed_failure`, and
  no-correct fails `min_success`. The legacy v1 policy still parses, so historical manifests
  keep validating. One shared helper, `complete_group_contract.meets_minima`, is used by the
  miner filter, the coordinator, the audit bridge and the release policy.
- Derail mask (`derail_mask`, `derail_window` = 32, `derail_threshold` = -3, `derail_tail` = 64).
  It applies only to truncated (`max_tokens`) rollouts with a negative advantage.
  - The derail index is the first position whose trailing 32-token mean of the sampled-token
    log-prob (trusted OLD) is below -3.
  - Only tokens 0..index+64 enter the loss.
  - Because the model is causal, the gradient pass and the KL reference run on that prefix
    only. OLD is re-scored on the same prefix, so the 1e-5 OLD/actor check is unchanged.
- Loss aggregation: `loss_agg_mode = seq-mean-token-sum-norm` with `loss_scale_factor` = 8192
  (the max response length). The loss is the token sum divided by (active rollouts × 8192),
  so every token weighs the same whatever the rollout length. It is microbatch-invariant
  because the global active-rollout count is passed in.
- `max_tasks` = 256 (all eligible; about 126 were last observed). Gradient accumulation is
  unchanged: one rollout per backward, with FP32 accumulator buffers.
- k3 KL: `kl_coefficient` = 0.02. The estimator is pinned upstream `kl_penalty_forward('low_var_kl')`.
  The reference is the cutover base `87de8304…` (epoch-0 input), not the epoch's input checkpoint.
  Each epoch makes exactly one optimizer step and OLD equals the actor, so KL to the epoch's
  input would be identically 0. Only a fixed anchor constrains cumulative drift.
  The reference must be pre-staged at `<trainer_state_root>/<run>/kl-reference/<id>/`.
  Its bytes are verified against the signed id.
- `sampler_correction`: `none` (default) or `sc-mis-replayed-support-v1`. The latter is score
  centering plus masked importance sampling on the top-k/top-p support, replayed from trusted
  prefill logits (the E42 objective). It is NOT enabled in the proposed config.

The legacy config document is byte-identical: unset extension fields are omitted, and the
version stays `affine-mimo-reference-grpo-v1`. Any extension produces
`affine-mimo-reference-grpo-v2`.
