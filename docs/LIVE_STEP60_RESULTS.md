# Fixed production step 60: independent evaluation found regression

The completed comparison against the genuine upstream 7B model found a
substantial regression at the preselected production step 60 checkpoint
`6134d074`. This measures that fixed checkpoint, not later live checkpoints.
Favorable scores on smaller, already exposed diagnostics did not establish
generalization to this independent cohort.

The 512 MATH test problems were reserved before evaluation, with known training
and previous evaluation identities excluded. Both models used the same prompts,
paired seeds, native grader, pinned FP32 evaluation runtime, temperature 0.7,
top-p 1 and 2,048-token output budget. Each task contributed one sampled response.
All 512 tasks remain in each denominator, including unresolved outcomes. The
two complete evaluation captures were authenticated and durably archived before
the comparison was read out.

| Model | Correct / 512 | Accuracy | Completed wrong | Unresolved | Budget caps |
|---|---:|---:|---:|---:|---:|
| Genuine upstream `2173b3e0` | 406 | 79.30% | 70 | 36 | 36 |
| Production step 60 `6134d074` | 335 | 65.43% | 108 | 69 | 72 |

The candidate gained 29 previously unsolved tasks and lost 100 previously
correct tasks: **−13.87 percentage points**, with the prespecified conservative
paired 95% interval **−20.17 to −7.35 points**. The improvement gate failed.
The interval subtracts exact 97.5% marginal Clopper–Pearson bounds for gain and
loss probabilities, with Bonferroni coverage. There were zero infrastructure
errors in the complete replacement evaluations.

## Stopping and length: shorter typical answers, more capped answers

Post-result descriptive analysis found a mixed change in response lengths.
The median fell from 500.5 to 165 tokens and the mean from 667.4 to 458.6;
responses of at most 32 tokens increased from zero to 57. At the same time,
budget caps doubled from 36 to 72 and terminal-EOS responses fell from 476 to
440. Token counts include EOS when present. All 512 paired prompt-token hashes
matched exactly.

Of the 100 lost correct tasks, 72 became wrong answers, with a median length
of 69.5 tokens, and 28 became unresolved at the output limit. Among those 72
wrong answers, 71 ended with EOS and one hit the cap. Across the full candidate
cohort, two capped responses were graded correct and one wrong; thus cap counts
and native unresolved counts are distinct measurements.

This is not explained solely by insufficient output length. It also does not
identify the cause of the shorter answers or justify accusing miners of fraud.
The [historical coverage and subject/level breakdown](LIVE_STEP60_STRATA.md)
also finds declines in well-represented groups; the causes remain unresolved.

The production training lineage is separate from the isolated
[positive-only SFT32 experiment](POSITIVE_SFT32_RESULTS.md), whose capped
responses decreased sharply. Their length patterns should not be conflated.

## Original failures and limits

The initial evaluation attempt on each model failed during resource monitoring
when a temporary compiler directory disappeared. Neither attempt completed a
task; both original failures and partial evidence remain archived. An explicitly
approved resource-only successor fixed that narrow scan race and ran one
replacement per model. Checkpoints, task identities, seeds, numerical evaluator
and grader remained fixed. The initial study is recorded as incomplete; this
result belongs to the complete replacement study. No outcome-dependent retry,
checkpoint selection or task omission occurred.

This result does not identify the harmful update, establish miner misconduct,
or authorize a production reset. Known local exposure exclusions cannot prove
absence of upstream pretraining or undocumented historical exposure. These 512
tasks are now exposed for future development and cannot be reused as fresh
confirmation of a later candidate. Continued network operation and hourly
weights are separate from the failed learning claim.

Aggregate comparison SHA256:
`8c92f44b7a1f00d3c8acf8c05630f38d4f9537ce01a49eb1918846b5b09559c0`.
Descriptive length/stopping report SHA256:
`a76ca6c469076df777d64c32921bc458cc97f33e072a6601b7b33b252f5aebd9`.
Original captures and failures have private full-GET archive verification.
Evaluation identities, individual outputs and private storage capabilities are
omitted from this public note. No model calls or production changes were made
to produce the descriptive analysis.
