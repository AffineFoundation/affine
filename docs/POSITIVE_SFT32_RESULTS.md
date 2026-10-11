## Positive-only SFT: the larger controlled study failed

The completed independent 512-problem comparison found a substantial regression
from supervised training on historical network positives. These were isolated
research models; neither endpoint has been promoted to the live network.

Both candidates started from the same original 7B checkpoint with fresh Adam.
They trained for exactly 32 updates, using the same 1,024 distinct training tasks
and two passes, a learning rate of 1e-6, FP32 master weights/moments/gradient
accumulation, and complete responses including EOS. One arm used all 9,994
qualified network positives; the other used one official training reference per
task. The latter is a diagnostic control, not a candidate chosen after seeing
results. Each fixed endpoint was evaluated on the same newly reserved 512 tasks,
paired seeds and 2,048-token output budget. All original results were retained;
there were no infrastructure errors or numerical reruns.

| Model | Correct / 512 | Accuracy | Completed wrong | Unresolved | Budget caps |
|---|---:|---:|---:|---:|---:|
| Original base | 405 | 79.10% | 76 | 31 | 32 |
| Network-positive SFT | 268 | 52.34% | 237 | 7 | 5 |
| Official-reference SFT | 389 | 75.98% | 94 | 29 | 30 |

Network-positive SFT was the sole preregistered primary comparison. It gained
22 previously incorrect tasks and lost 159 previously correct tasks: **−26.76
percentage points**, with a conservative paired 95% interval of **−33.33 to
−19.77 points**. The primary success gate failed. The official-reference arm
had 30 gains and 46 losses: −3.13 points, paired 95% interval −8.46 to +2.26.
The additional preregistered 97.5% intervals for the two base comparisons were
−34.04 to −18.97 and −9.04 to +2.86 points, respectively. The descriptive network
versus reference difference was −23.63 points, 95% interval −30.30 to −16.60;
it is not a separate selection or promotion gate. Intervals use differences of
exact marginal Clopper–Pearson bounds with Bonferroni coverage.

The network-positive model learned its training targets: positive training NLL
fell from 0.486976 to 0.140064. That fit did not transfer. Its median evaluation
response shrank from 503.5 to 51 tokens; mean length fell from 652.7 to 143.8.
Of the 159 lost correct tasks, 156 became completed wrong answers and only three
became unresolved. This is not principally a shortage of output tokens. The
reference model retained a median of 475 tokens and training NLL fell from
1.075049 to 0.799114.

A training-only descriptive analysis identifies a plausible objective/data
interaction. The loss averages tokens within each response, then responses
within each task. Consequently, shorter responses give each individual target
token, including EOS, greater weight. Network responses of at most 32 tokens
account for 12.91% of task-balanced objective mass but 52.33% of its terminal-EOS
coefficient. Overall EOS coefficient is 2.32 times the reference corpus's.
These are algebraic loss weights, not measured gradients or proof of causation.
A strict boxed-answer-only heuristic matches just 94 of 9,994 network responses;
shortness alone does not establish bad reasoning or cheating.

The next bounded tests should distinguish loss normalization from data quality
and selected-response bias. Per-task token-weighted normalization is a direct
ablation on the same frozen data, although it only reduces this corpus's EOS
coefficient by 20.55%. Fresh groups with every prescribed attempt retained are
another direction, after the new RL backward/optimizer path passes a bounded
mechanics check. No new numerical study or live objective change follows from
this report alone. The network/reference comparison also differs in response
count and approximately 9.4-fold token exposure, so it cannot isolate a single
cause.

Original aggregate comparison SHA256:
`493973e2855ebd50d6bd4ab35cf6e274cc0aad4d594ec583cf9726967e79ac3e`.
The full original captures and models are privately archived with verified GET
readback. Evaluation identities and individual outputs are omitted here.
