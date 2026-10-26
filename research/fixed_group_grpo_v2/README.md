# Private complete-group learning comparison

The signed private comparison started on two dedicated H200s on
2026-10-12 at 21:47:55 UTC. It does not change production. This is a separate
successor to the resource-incomplete v1 study, whose frozen sources and original
data remain intact. There is no learning result yet.

Frozen dose: 16 updates, target4/minimum2 complete mixed task groups/update,
maximum48 predeclared candidate tasks/update, N8 unchanged. BF16 actual behavior
probabilities stay in the PPO denominator. Eligibility requires at least one
completed correct and one completed wrong response; unresolved trajectories stay
in original group statistics and are wholly masked from learning. Every group
retains all canonical nonces. No replacement by recomputed training probabilities.

Numerical usability is assessed at the update level with finite/identity checks,
recorded drift distribution and sparse-tail guards; this is separate from proof
acceptance or cheating attribution. Frozen v1 per-token limits are not modified.

The candidate uses group-relative advantages, clipped token ratios against
actual generation probabilities and differentiable sampled k3 KL to the frozen
base model. The control uses all completed correct/wrong preference pairs within
each complete group. Both start from upstream2173 with fresh FP32-master AdamW,
LR1e-6 and BF16 forward/backward. The task schedule is fixed before outcomes;
each arm subsequently samples its own updated policy.

Native grading reuses authenticated initialization through
`research.native_math_reuse_v2`, while each answer is graded in a fresh isolated
CPU child. Reference failures consume their original task slot. Infrastructure
errors abort the original experiment and are never counted as wrong answers.

Runtime scopes, preflight receipts, source pins, phase supervision and immutable
capture records are in
`state/root-audits/20261012-fixed-group-grpo-v1/successor-v2/`. The private worker
requires a ROOT-signed scope and cannot publish network checkpoints or weights.
No automatic numerical retries or production promotion are enabled.

The original development128 baseline is reused only with unchanged evaluator
bytes, model, prompts, sampler and random-draw identity. Each complete update16
endpoint is evaluated once; a passing development screen still requires the
independent frozen confirmation512 comparison and production integration review.
Group size may change in a future predeclared experiment; it cannot change after
seeing outcomes within this one.
