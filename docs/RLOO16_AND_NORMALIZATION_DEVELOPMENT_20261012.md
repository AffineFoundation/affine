# RLOO16 and loss normalization: development results, confirmation pending

As of 2026-10-12, the completed development comparisons do **not** establish a
learning improvement. Fresh-group RLOO finished one correct answer above the
original base. Changing the positive-SFT loss normalization recovered two
answers relative to its control, while both SFT models remained below base.
None of these research endpoints has been promoted to production.

All four models were compared on the same fixed 116 TRAIN problems, using
identical prompts, paired seeds, native grading, FP32 evaluation, temperature
0.7, top-p 1 and a 2,048-token output budget. These 116 problems are the
prospectively fixed subset of a 128-problem development cohort after excluding
12 SFT training overlaps. All original cases, including unresolved answers,
remain in their denominators. Every original capture passed full private GET
readback before the shared baseline was unsealed. Actual prompt-token, seed,
source/runtime and tokenizer-behavior comparisons passed; all four originals
had zero infrastructure errors.

| Model | Correct / 116 | Accuracy | Completed wrong | Unresolved | Budget caps |
|---|---:|---:|---:|---:|---:|
| Original base | 99 | 85.34% | 14 | 3 | 4 |
| Response-mean positive SFT32 | 83 | 71.55% | 30 | 3 | 3 |
| Within-task token-pooled SFT32 | 85 | 73.28% | 29 | 2 | 3 |
| Fresh-group RLOO16 | 100 | 86.21% | 12 | 4 | 4 |

The SFT normalization ablation used the same 1,024 tasks, 9,994 positive
responses, two passes and 32 fixed updates as its control. It pooled token loss
within each task instead of averaging each response's mean token loss. Its
primary paired difference was **+1.72 percentage points**: three gains, one
loss, conservative 95% interval **−4.96 to +8.14 points**, one-sided exact
McNemar p = 0.3125. This does not establish that normalization fixes the earlier
[independent SFT regression](POSITIVE_SFT32_RESULTS.md).

RLOO used 256 separately selected TRAIN tasks, eight fresh unselected attempts
per task and 16 fixed updates from fresh Adam. On common116 it had two gains
and one loss versus base: **+0.86 points**, conservative 95% interval **−5.23 to
+6.81**, p = 0.5. Its original full128 result is retained separately:
**108 versus 107**, +0.78 points, 95% interval −4.75 to +6.19, p = 0.5.
The cross-recipe comparisons are descriptive: training data, objective and
update budget differ. These intervals are individual paired intervals, not a
familywise claim that a winning method has been established.

The original RLOO training record retained all 2,048 prescribed attempts;
892 (43.6%) were unresolved/capped, and 84 of 256 groups had equal binary
rewards. This may limit useful reward contrast, but it does not diagnose the
cause of the small development difference. Tasks and checkpoints changed
across training groups, so these counts are not a fixed-policy pass-rate or
public K4/L4 quota-feasibility estimate.

A rule frozen before the shared development readout selected only the original
final RLOO16 endpoint for new confirmation. Its paired comparison against the
original base is now running on a newly reserved, independently qualified
512-problem MATH test cohort. Tasks, seeds, checkpoints, grading and output
budget are fixed; both original outcomes will be retained. Scores remain
sealed until both complete captures are durable. **Confirmation is pending;
there is no promotion or production objective change.**

Paired intervals subtract exact marginal Clopper–Pearson bounds with Bonferroni
coverage. Aggregate common116 report SHA256:
`9d62fecae385cf2ab6a543ab1a5223b00d84a578f52db59e2fb5b801c13be36a`.
Comparability report SHA256:
`2cc4a102db6edc291c885e71cd84ac316824844dae5c4d5d0d1b86d5181f63f2`.
Individual task identities, responses and private storage capabilities are
omitted. Known local exposure exclusions cannot rule out upstream pretraining
or undocumented historical exposure.
