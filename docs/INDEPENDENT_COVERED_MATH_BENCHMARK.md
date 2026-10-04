# Precommitted independent MATH comparison

The earlier 128-task comparison found 88 baseline successes and 89 trained
successes, with six paired gains and five losses (two-sided exact McNemar p=1).
It did not demonstrate improvement. A new cohort is precommitted before the
full-coverage training handoff to test the next learning stage independently.

The new cohort contains 200 distinct reserved MATH problems. Both task indices
and exact problem hashes are disjoint from mining, the fixed 32-task diagnostic
and the previous 128-task benchmark. The original dataset hash, problems, seeds,
free autoregressive evaluation harness and scientific source files are frozen
in an operator-signed plan. Its envelope SHA256 is
`8d169fd8690f2d8355bc9e4ca3724cce721cece6d579510f83284998d0e7b497`.
Problem identities and seeds remain in the private plan until final disclosure.
This excludes known subnet-training overlap; it does not establish absence of
pretraining contamination or semantically equivalent problems elsewhere.

The baseline is checkpoint
`660098235e1c7bce048679292f26f5255834d0a0abbee8df5dfefbf2fa3147da`,
after twelve public training updates. A retained independent H100 runs thirteen
bounded shards through signed GPU jobs, full model/TOPLOC verification and the
original native grader. Each shard retains its original request and report;
completion requires all 200 original rows without grader failures. The original
baseline runner and child have been observed live; the run is not complete.

The comparison checkpoint is predetermined: the checkpoint after the first three
nonempty, completed public epochs using
`bf16-full-adamw-covered-fixed-reference-v3` after the cohort precommit. Selection
must not depend on this benchmark's scores. The comparison uses the same task
indices, seeds, harness and scientific computation pins. Intervening legacy
training may occur; a gain would not isolate a causal effect of the new policy.

Report every paired outcome, both success totals, gains, losses and the exact
two-sided McNemar test. A positive paired delta and p<0.05 are the predeclared
test for improvement in this comparison; one comparison alone does not establish
steady long-term learning. No gain is claimed before both complete original runs
and independent source, checkpoint, request, report and outcome review.

Evidence is retained under
`state/live-math-launch-preparation-v1/continuous-upgrade-f590-new9-v1/independent-covered-math200-precommit-v1`.
The benchmark does not change a public mining epoch or submit chain weights.

The first baseline attempt stopped after 80 verified rows: a generated null
character caused the original native grader's subprocess argv transport to
raise `ValueError: embedded null byte`. Those original jobs, failed process
waits and first 80 diagnostic rows are preserved. An exception is not counted
as a model failure and no task is removed from the cohort.

A separately signed transport-only amendment binds source
`3bacecbf006f5f1aa5fc274f4a68a444e91183e6078990bc00e9f138a2c1eb08`
and its native environment fingerprint. JSON argument transport preserves the
exact model text and leaves the grading predicate unchanged. All model,
generation, GPU, harness and proof computation module bytes are unchanged from
the precommit. Normal/null-text native controls and the repaired source's H200
training/proof control passed independent review before applying the amendment.

All 200 baseline problems are being run again under that amended transport,
with the original checkpoint, indices, seeds, harness, comparison-checkpoint
selection rule and statistical test unchanged. The complete amended baseline
must pass independent review; the first 80 old rows are not mixed into its
results. The comparison must use the same amended transport. This amendment
is a disclosed infrastructure repair, not selection of a different cohort or
checkpoint based on benchmark scores. See [the transport qualification](MATH_GRADER_TRANSPORT.md).
