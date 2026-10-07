# Same-parent training stability review

Prospective research only; this review authorizes no job, source, reward,
quota, sampler, optimizer or production change. Read with
[the paired-sample plan](PAIRED_SAMPLE_STABILITY_PLAN.md). CPU algebra and
source review do not measure learning or replace the two real branch trials.

## What the current update actually does

The frozen f213 trainer and current main have identical
`task_normalized_training.py`, `epoch_optimizer.py` and
`persistent_cpu_adamw.py` bytes. Their SHA256 values respectively are
`2d86925d9fb1533793a942734da524d445f897e355da4b968b38624f9880eda8`,
`d0ef5dc19ddbf109c015b1dc7d4ee1327e30cc3e7c63dc96e0c7c45c82d5c8b4`, and
`f0624e2955525ac2cc2a666c7d45365a6c53b1d27513c6a5bc33277960c6ad35`.
The current main sampler/model have evolved; their bytes must not be used to
explain historical frozen f213 execution.

For each rollout, the trainer sums selected-token log probabilities over all
turns and divides by all output tokens. EOS, when emitted, contributes to this
mean. There is no censoring mask, first-invalid-token cutoff or special
termination loss. A cap-length output without EOS contributes its entire tail.
The f213 sampler stops on tokenizer EOS only (151645), not the other EOS ID in
the generation configuration. An EOS at the final allowed position is complete.
A model-vocabulary-valid token outside the tokenizer mapping is an important
quality diagnostic, not by itself proof of a sampler violation: f213's token
validator accepts the model width, and sampling authenticity is a separate gate.

For pair margin m and fixed epoch-input reference r, the loss is
`-log sigmoid(0.1 * (m-r))`. At the input checkpoint, r=m for every pair. Thus
initial loss is log(2), and d(loss)/dm is -0.05 for both an initial margin near
zero and a margin above 10. A large pre-existing preference is not an easy-pair
filter. Each output token's explicit coefficient also divides by its own
rollout length; short positives and long negatives have different per-token
coefficients. This does not determine parameter-gradient directions or prove
that longer negatives dominate the gradient.

Pairs are averaged within each task, then tasks within each update group.
Two pairs for a task do not double its weight. With one update, all selected
tasks contribute to that one update; with several updates the scheduler
partitions tasks, and later groups see already-updated parameters. Increasing
update count is therefore not a clean test of sample quota.

AdamW preserves FP32 master parameters, first and second moments and the global
counter. LR is 1e-5, betas .9/.999, epsilon 1e-8, weight decay .01, beta .1 and
clip limit 1.0. The epoch reference refreshes, but Adam state does not reset.
There is no fixed-base KL anchor. The authenticated E38 norm .1088867 is below
the clip limit; that report does not support a clipping explanation for its
regression. Finite losses and changed parameters establish an effective update,
not generalization.

## Evidence and competing explanations

The exact E37 join binds original documents/batches and both pair-digest forms:
91/256 negatives are capped without EOS with margins above 5; 89 of those
batches have accepted audits and two have TOPLOC rejection. Those 89 have
FP32 mean margin 10.915 versus BF16 10.902. Their final 512 tokens average
-11.856 nats and 82 contain tokenizer-unmapped tokens after tail onset.
This weakens a BF16-only explanation of this tail. It does not authenticate
historical per-position CDF execution anew, identify intent or establish cause.

The mechanisms still competing are:

- Censored/low-quality negatives train against a failure tail rather than a
  completed wrong solution; their initial loss coefficient remains nonzero.
- Refreshing the reference removes a persistent initial-model anchor. Small
  successful local preferences can accumulate drift across epochs.
- Task selection and coverage change with supply, cap rates and miner mix.
  A comparison with unequal task counts would confound negative quality.
- Larger per-task sample count may reduce variance, but may also reduce distinct
  tasks per mining budget. The current optimizer already averages many tasks.
- Historical numerical/sampling acceptance limitations may admit problematic
  trajectories. Do not convert this possibility into a blanket fraud finding.

CP29 versus CP28 is a matched 76/128 versus 78/128 (four gains, six losses);
CP23 was 90/128. This is regression evidence, not a causal attribution to any
one mechanism. SAME128 is now a repeatedly inspected diagnostic; reserve the
untouched wider cohort for confirmation, and do not choose arms using its answers.

## Next controlled comparison

Finish the already prepared CP24 eight-task negative-quality comparison before
starting another experiment. Restore the exact same CP24 inference model and
Adam24 descriptor/full 23 shards independently for each branch. Both branches
advance 24 to 25 exactly once; neither continues from the other's output.

Use the eight predeclared audited matched tasks and the same original positive
trajectory per task in both branches. Baseline uses the original capped wrong
negative; the comparison uses the original completed-EOS wrong negative. Keep
all original labels, trajectories, sampler receipts and proof artifacts intact.
Each task has weight 1/8, each arm has exactly eight tasks and one pair per task.
Freeze identities before training. Audit-negative exclusion must be identical
in both arms; no BF16 teacher-forward is substituted for an FP32 inference audit.

Hold source f213/177, parent model and Adam state, LR/beta/dropout/objective,
reference rule, update count, ordering, resource policy and independent 4db
SAME128 template/seeds/native grader fixed. The intervention changes negative
termination, length and content together: it tests the *completed-negative
selection policy*, not a pure EOS-token effect. Document those residual
confounds rather than truncating, repairing or relabeling an original output.

Before and after the step, report task-level margins and positive/negative
mean log probability separately, lengths, EOS/cap rates, tokenizer-unmapped
counts, tail-position diagnostics, gradient norm, clipping, master/BF16 changes,
optimizer counter and actual restore/train/export costs. On SAME128 report
paired gains/losses and parent-to-arm and arm-to-arm comparisons, not only score
or training loss. Keep original baseline CP24 report provenance.

If the matched eight-task gate is unavailable or either branch lacks original
wait/output/full durable ACK, report the test incomplete; do not replace tasks,
redraw or reset the optimizer. One small branch test is a mechanism screen,
not sufficient evidence for production convergence.

Only after this result should a separate nested K1/L1 versus K2/L2 trial hold
negative-quality strata and task count fixed. Reference anchoring is a later
separate arm requiring its own signed research objective; do not combine it with
quota, termination selection or a learning-rate change.

## Private evidence selectors

Immutable local receipts reviewed for the factual statements above (not copied
into the public tree) have full SHA256:

- E37 document/token/audit join: `8ebc280522f4a8f76d86d1c0bdb8a873a437eaf213150b08b89e42a7aade739a`;
  original trainer report: `837b6a477208c4a1beec8c44dd0e2dac82372c2929fe9e2839ec2228c751f567`.
- E37 tail transition: `10c1fac543c911845d96d5d717bb3c9f76e3a8507140e47040f7aef127210ade`.
- E37 effective audit context: `272fb9542c4d349579ff049ebda6b0b7c807beb01add66aa579aee5faa8a5031`.
- CP29/CP28 SAME128 matched original-ACK comparison:
  `5a758bf737ae6f5c8b2de4b2160cd7d3b968986103529b200c1d63d155357f4b`.

These are provenance selectors, not new signatures, GPU replays or authorizations.
