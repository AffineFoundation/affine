# Training stability and convergence review

This is an evidence review and prospective plan, not a change to the active
mining contract. Review scope: the immutable forced-sampling candidate
`e415623e5a8017133ff7bef3925b724441862f34b6d034ed0deb6b731a26e657`,
its `bf16-full-adamw-covered-fixed-reference-v3` trainer, and retained local
control records. No remote job, service, checkpoint or chain transaction was
changed by this review.

## What the implementation establishes

The worker downloads only signed frozen submissions, recomputes inference and
native environment outcomes, and supplies only fully audited positive/negative
pairs to training. Forced sampling is required in the new candidate's audit.
The training reference is the exact epoch input checkpoint, computed before
any optimizer update. Prompt tokens supply context but do not receive a direct
target loss; only generated output tokens enter the preference score.

The covered policy removes exact audited pair clones and deterministically
partitions every remaining pair among optimizer groups. Each group's mean loss
is backpropagated one pair at a time, then clipped and updated. This fixes the
legacy v2 policy's selection of only the first three pairs for three updates.
With fewer pairs than updates, pairs repeat; a report must distinguish unique
pair coverage from optimizer-step count.

The genuine forced-sampling cross-host control fully reaudited one real pair
and performed one update. All 339 parameter tensors received gradients; the
parameter-value digest changed, and the independently loaded successor passed
proof verification. The earlier five-pair controls cover three updates and all
five pairs. These are update/proof controls, not evidence of held-out learning.

## Numerical risk demonstrated locally

The current optimizer uses BF16 parameters, BF16 Adam moments, learning rate
`1e-5`, `foreach=False`, and AdamW's default weight decay. It creates a new
optimizer for every epoch. It has no FP32 parameter master or retained update
residual. Receiving a gradient does not ensure a representable parameter change.

An actual CPU PyTorch control reproducing that optimizer performed three steps
with gradients of one at parameter values `1.0`, `0.1`, `0.02`, `0.01` and
`0.005`. Every BF16 value remained exactly unchanged; FP32 controls moved about
`3e-5`. Values at `0.002` and smaller did move in the BF16 control. This proves
the failure mode exists, but does not measure its prevalence in the full model
or prove that current training cannot learn at all.

Simply keeping FP32 state within an epoch can also lose progress when its
three-step result is rounded to BF16 and reloaded. A second control shows that
four such epochs leave a `0.02` value unchanged, while a persistent FP32 master
over the same twelve steps eventually produces a changed BF16 export.

Reproduce the three focused controls with:

```bash
.venv/bin/python -B -m unittest discover -s tests -p test_training_precision_diagnostics.py
```

Before claiming full-model learning, measure actual changed elements per tensor,
parameter-delta norms, and preference-margin changes. A changed overall hash
and 339 non-null gradients are insufficient by themselves.

## Next contract, in order of importance

1. Get a real new-contract epoch through upload, independent forced-sampling
   audit, score attribution, training and publication. Preserve failed attempts
   and exact source/seed/checkpoint lineage. Do not weaken replay checks to obtain
   throughput or nonempty epochs.
2. Qualify a precision-preserving trainer. Preserve an FP32 master or a tested
   compensation residual across epochs, and bind its durable state to the exact
   BF16 inference checkpoint published for miners. Preserve optimizer moments
   and counters or explicitly document and test a deliberate reset. Test real
   GPU memory admission; full FP32 buffers must not be assumed to fit an H200.
   CPU-offloaded state is a correctness-first fallback. Changing this policy
   requires a new immutable source and signed future-epoch contract.
3. Add audit-traceable learning diagnostics: selected and unique pair counts,
   actual pair/token gradient coverage, pre/post chosen and rejected mean log
   probabilities, preference margin change on the same verified pairs, gradient
   norm/clip fraction, per-tensor changed-element fraction and parameter-delta
   norms. Evaluate a bounded training-only diagnostic cohort with independent
   reload, so a decrease in reported loss cannot be a stale in-memory artifact.
4. Preserve the existing frozen held-out comparison and run fresh held-out
   evaluations at predetermined checkpoints. Only then change batch caps or
   consider reducing expensive audits.

The current objective is a reference-relative preference loss on **mean**
generated-token log probability with beta `0.1`. It is not standard sequence-sum
DPO or an unbiased policy-gradient objective. Long traces receive smaller
per-token loss weight, and a failed trace can contain useful reasoning before
its mistake. Success/failure searching also selects a subset of on-policy
draws. These are disclosed learning choices, not sampling-proof failures.
Do not silently substitute sum loss, increase learning rate, add teacher/SFT
traces or change sampling to address slow improvement. Compare any candidate
on frozen training controls and reserved evaluation tasks under a new contract.

Reward deduplication and training deduplication currently differ. Reward points
are unique by environment and task index across miners. The trainer accepts all
fully audited pairs and collapses only exact pair copies, so distinct valid
pairs for one task from many identities can receive many gradient shares even
when that duplicated task earns zero reward points. Forced sampling does not
remove this task-concentration risk. Decide explicitly whether to average
verified pairs within each task and give tasks equal training weight, or impose
a bounded per-task training quota. Publish and adversarially test that separate
training rule; do not silently reinterpret historical rewards or drop audited
data from an existing training request.

Also distinguish epoch-local reference regularization from protection against
long-term model drift. The current reference resets to the input checkpoint
each epoch; it is not a fixed original-model KL constraint. Keep a frozen
baseline and monitor held-out regressions and sampling entropy while comparing
candidate trainers. A lower preference loss on selected pairs alone can reflect
fitting those traces without better independent problem solving.

## Convergence measurement

The fixed 32-task diagnostic is useful for catching regressions, but it has
been inspected repeatedly and is too small to establish convergence. The
previous 128-task paired comparison had six gains and five losses, exact
two-sided McNemar p=1. The independent precommitted 200-task baseline is
131/200 at checkpoint `66009823...`, twelve old updates. Its comparison
checkpoint is fixed in advance: after the first three nonempty completed public
covered-policy epochs. The cutoff must not be selected using benchmark score.
The baseline predates the current fifteen-update input checkpoint, so even an
improvement will not isolate a causal effect of the new policy.

For that original comparison, preserve the original task hashes, seeds,
autoregressive harness and repaired native transport/source contract. Evaluate
the selected new checkpoint under that approved evaluation source; changing
the training/mining source does not authorize changing the comparison's
evaluation semantics. Report every paired outcome, gains, losses, errors and
exact McNemar p-value. Infrastructure/grader errors are not model failures, and
partial runs cannot replace the original complete cohort.

For ongoing stability, precommit checkpoints by completed nonempty epoch count,
not wall clock or favorable diagnostic scores. Use a separate reserved cohort
to check the current pre-cutover checkpoint versus later checkpoints under one
unchanged evaluation harness. Freeze that plan before inspecting its outcomes;
do not replace or select the existing 200-task benchmark retrospectively.
Never train on evaluation tasks, exact duplicate problem texts, evaluator
responses or failure recoveries. Keep the existing 750 reserved task indices
excluded from every mining epoch. Index/exact-text disjointness does not rule
out semantically duplicate tasks or pretraining contamination.

Evidence of convergence requires reproducible gains on a complete independent
cohort across multiple predetermined checkpoints, with uncertainty and all
regressions retained, not a single positive training margin or one improved
32-task score. Retain grader integrity checks, negative examples at the output
limit, nonfinite refusals and independent reload controls throughout.

## Scale only after these gates

After honest new-contract batches earn scores and the precision-preserving
trainer learns verified tasks without material held-out regression, measure
verifier demand, actual audit latency and epoch bottlenecks. A larger quota
needs byte/token caps and bounded audit allocation. Reward uninspected batches
only under an explicitly designed and adversarially tested estimator/penalty
policy; the current guarantee remains fully-audited-only. Publish contract and
client changes together with the exact public `/llms.txt` before miners adopt
the next epoch.

No finite audit or test set proves a subnet is exploit-free. Completion should
state the tested threat model and residual risks, including task leakage,
cross-hardware replay, verifier compromise, repeated-failure selection and
duplicate manipulation. Statistical convergence and exploit resistance are
separate acceptance gates.
