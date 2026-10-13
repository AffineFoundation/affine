# First two public MiMo updates

These are observations from finalized epochs 189 and 190 of
`affine-mimo9b-grpo-20261013`, not a claim of convergence. The issued manifests,
training inputs, reports and scores retain their original rules.

| Measurement | Epoch 189 | Epoch 190 |
| --- | ---: | ---: |
| Submitted task groups | 42 | 95 |
| Distinct submitted tasks | 42 | 88 |
| Training tasks / trajectories | 32 / 256 | 32 / 256 |
| Correct / completed wrong / unresolved training trajectories | 162 / 45 / 49 | 149 / 61 / 46 |
| Active training trajectories | 207 | 210 |
| Distinct contributing training miners | 26 | 28 |
| Preclip gradient norm | 0.54437 | 0.44146 |
| Training computation, including optimizer | 480.40 s | 520.26 s |
| Full epoch, including collection and publication | 70.27 min | 68.23 min |

There were 63 distinct trained tasks across the 64 task slots. Early training is
therefore not repeatedly consuming one tiny task subset. This says little about
longer-term coverage. The second epoch excluded 14 groups on seven collided tasks
and another 49 otherwise eligible groups because the update budget is 32 tasks.
The miner submission cap and trainer task budget are separate limits.

Both updates use LR `1e-6`, one PPO epoch, FP32 gradient accumulation and CPU
FP32 AdamW masters/moments. KL and entropy coefficients are explicitly zero.
Neither update reached the gradient clipping threshold of 1. A negative reported
policy objective is not held-out loss and is not itself evidence of improvement.

## Measured data-quality issue

Most unresolved training outputs ended with EOS, rather than exhausting the
token budget: 42 of 49 in the first update and 43 of 46 in the second. Replaying
the original tokens through a narrow *prospective* completion parser recovers
30 and 27 respectively. It allows an optional terminal period and fixes attachment
of math delimiters around a single final box. The original native math grader then
labels these as 27 correct / 3 wrong and 23 correct / 4 wrong.

Every previously completed verdict and grading string is preserved by this
diagnostic. Budget-capped answers, repeated boxes, invalid channels and trailing
units are not accepted by the proposed rule. This parser is not deployed. A
grading change requires its own qualified, versioned cutover and matched baseline.

## What we know about learning

The matched public fixed32 result after the first update is officially 18 to 19
correct. Applying the same narrow formatting diagnostic to both checkpoints gives
22 to 22. Neither result establishes a trend; see the
[matched evaluation diagnostic](MIMO_FIRST_PUBLIC_EVAL_DIAGNOSTIC.md).

The current objective centers rewards across all eight attempts, then masks
unresolved trajectories. When unresolved responses exist, the remaining active
advantages no longer have zero unweighted sum. The pinned reference implementation
also supports a valid-only centering option. Comparing those options is a proposed
controlled experiment, not a demonstrated fix. Token-length weighting further
affects their relative contributions. Changing the parser, centering, task budget
and learning rate together would obscure which change helped.

At the authenticated 45-audit-report snapshot, 27 of the 64 trained groups had
accepted audit reports, four had rejected reports, and 33 had not been audited.
Three of those four rejections were independently traced to cached sampling
disagreements; the fourth had not yet been localized. This warrants continued
runtime and data-quality investigation, but does not establish miner intent or
the rejection rate of unaudited groups. Training from eligible unaudited groups
remains the published contract.

Next measurements are the next matched fixed32 result and a private comparison
on 128 preselected held-out tasks disjoint from mining and fixed32. That larger
comparison uses the original grading/runtime on both checkpoints; no relaxed
updated score is compared with a strict baseline. Results will be reported when
the actual evaluations finish.
