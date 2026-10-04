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
