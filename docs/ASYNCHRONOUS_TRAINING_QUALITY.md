# Prospective asynchronous training-quality monitor

This code is not deployed and does not gate audits, reverify samples, alter E14,
or select a new source. `subnet.training_quality_monitor` compares original
ROOT-authenticated before/after evaluation publications plus an explicit
`authenticated-training-quality-input-v1` operator summary admitted from the
original training job and committed optimizer descriptor. The adapter that
creates this summary must preserve the original report hash, request/source
pins and exact parent/output state descriptors; do not admit an untrusted
standalone training assertion. Summary fields bind epoch/checkpoints, actual
optimizer steps, state hashes, changed inference weights, and original finite
loss/gradient/margin diagnostics. Generic historical `verified_pairs` names do
not change UNAUDITED assurance.

Only complete same-task/grader/dataset/harness/runtime comparisons qualify.
Missing evaluator results or grader infrastructure errors are unknown, not
regressions. The default uses at least128 tasks and nonoverlapping95% Wilson
intervals plus a5% drop; a20% confirmed drop recommends a hold immediately,
and three successive confirmed drops recommend a hold. Fixed32 results stay
informational. These conservative heuristics are not a formal sequential
multiple-testing guarantee or convergence proof. Pairwise task transitions
would improve sensitivity when raw authenticated per-task outcomes are exposed.

Monitor records and hold documents must be signed, immutable and idempotent.
`reconcile` checks unbroken checkpoint AND optimizer-state lineage and dedupes
same evidence. A future controller integration should consult the signed hold
before dispatching the next training job and before authority publication;
in-flight work can finish privately but must not silently commit a held branch.
This consultation does not wait for audits or for every evaluation. No quality
hold may rewrite already committed checkpoints. Operator review of the original
parent checkpoint AND matching optimizer/genesis lineage is mandatory before
any rollback; never reset or mix old weights with new optimizer moments.

`evaluation_indices` provides a prospective deterministic rotating subset of
reserved tasks (128/update default), plus full reserved evaluation every24
updates. Both before and after use the SAME rotation, grader, seeds and budget.
Keep the original32 diagnostic separately comparable. Never feed the750 reserved
tasks into mining/training. Run expanded checks on the independent evaluator,
rate-limit backlog by actual capacity, and publish unknown/incomplete status
instead of a partial score. This plan needs an explicit future signed evaluation
policy and controller integration after the first real update is validated.

Controls cover source-independent numeric comparison, signatures, exact task
comparability, incomplete evaluations, small32 noise, confirmed severe/repeated
regression, lineage breaks, evidence deduplication, and leakage-free rotation.
No production job or GPU work is launched by the tests.
