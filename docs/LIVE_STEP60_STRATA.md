# Historical coverage and the released step-60 regression

The [released step-60 comparison](LIVE_STEP60_RESULTS.md) scored 335/512 against
406/512 for the original base model: 29 gained answers and 100 lost answers.
This retrospective breakdown uses that same completed comparison. It makes no
new model calls and does not inspect pending learning-method experiments.

The exact training history before that checkpoint is epochs 91–150: **24,865
task-update exposures across 2,793 distinct tasks out of 6,746**. An exposure
counts a task once in each update containing it. It is not a token count,
gradient weight or task success probability. These historical counts differ
from the later coverage trial in epochs 154–156, which cannot explain an
earlier checkpoint's performance.

The 512 evaluation tasks and 6,746 mining tasks have disjoint identities. This
compares their subject and level groups, not training on the evaluated tasks.

| Dataset level | Distinct trained / available | Task-update exposures | Exposure relative to universe share | Evaluation gains / losses | Accuracy change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 88 / 510 | 444 | 0.236× | 1 / 7 | −12.77 pp |
| 2 | 366 / 1,208 | 2,598 | 0.583× | 4 / 16 | −11.88 pp |
| 3 | 570 / 1,431 | 4,428 | 0.840× | 6 / 31 | −21.74 pp |
| 4 | 718 / 1,536 | 6,186 | 1.093× | 9 / 30 | −15.56 pp |
| 5 | 1,050 / 2,059 | 11,204 | 1.476× | 9 / 16 | −6.14 pp |

Two mining tasks have unknown levels; one was trained in five updates. There
are no unknown-level tasks in the evaluation cohort. Evaluation denominators
for levels 1–5 are 47, 101, 115, 135 and 114. All unresolved responses remain in
those denominators. Dataset levels do not measure difficulty for this model.

The imbalance is substantial: Level 1 contributed only 1.79% of exposures
despite comprising 7.56% of the mining universe, while Level 5 contributed
45.06% against 30.52%. Nevertheless, the greatest net decline was at Level 3,
not the most neglected level. Baseline performance also differs: Level 1 lost
7 of its 45 previously correct answers; the other levels lost 93 of 361.

The subject breakdown likewise does not follow a simple neglect pattern.
Algebra, Prealgebra and Precalculus were underrepresented in both distinct
training coverage and repeated exposures. Together they declined 10.15 points,
across 266 evaluation tasks, versus 17.89 points across the other 246. The largest
subject declines were Counting & Probability (−27.27 points on 44 tasks) and
Number Theory (−26.67 points on 45 tasks), which received
1.609× and 1.053× their universe share of task-update exposures.

Stopping behavior is a separate concern. Of the 100 lost correct answers,
**71 became wrong answers ending with EOS**, one became a capped wrong answer,
and 28 became capped unresolved answers. The 72 wrong answers had a median
length of 69.5 tokens. Raising the output budget alone does not address most
recorded losses; EOS termination also does not establish sound reasoning.

These are descriptive, post hoc groups with different starting accuracies and
one sampled answer per task. They do not identify a harmful update, prove a
coverage effect, or establish miner misconduct. Coverage, response quality and
the training objective remain separate hypotheses. The prospective coverage
test and controlled learning-method experiments continue separately; neither
this analysis nor broader coverage establishes improved learning.

Evidence SHA256:

- Released paired comparison: `8c92f44b7a1f00d3c8acf8c05630f38d4f9537ce01a49eb1918846b5b09559c0`
- Frozen training census: `1546c3f5fdc63100bfb307896117e80b5f1f1a2cce682e49ac65424803daca74`
- Exact pre-step60 exposure analysis: `fa14aabeea102909a5246b1cc3736d7352a6449d8198dd0428bd84bed3b85352`
- Released outcome strata: `ff197a2ff36a32bc132703ed408ea77f9d1beecd3ea4fb291107c63078d07ddf`

All 60 historical signed manifests, completion records and training jobs were
authenticated, with report hashes and checkpoint/optimizer continuity checked.
Training attribution comes from successful reports bound to those records,
not an independent proof of physical execution. No task text, reference answer,
generated response, private URL or credential is included here.
