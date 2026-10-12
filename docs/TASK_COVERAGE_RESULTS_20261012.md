# Closed-epoch task coverage: through epoch 156

**Latest status, October 12 at 05:00 UTC:** the fixed 154–156 cooldown trial is
complete. A trainer-side priority for tasks absent from the previous twelve
training epochs is scheduled from epoch 158. It has not yet completed a live
training update. The earlier sections below preserve the evidence and decisions
available at each stage; the final section contains the completed trial results.

The network is repeatedly training on a restricted portion of the available
MATH tasks. This is a plausible contributor to weak generalization, not proof
of its cause or evidence of miner misconduct.

Across the twelve completed epochs, the 6,746-task mining pool had:

| Stage | Distinct tasks |
| --- | ---: |
| At least one completed submission commitment | 2,989 |
| Captured for eligibility processing | 2,791 |
| Reported as actually trained | 2,484 |

The other 3,757 tasks had no observed completed commitment. Failed or incomplete
search attempts are not fully recorded, so absence cannot distinguish unvisited
tasks from unsuccessful searches. Another 505 submitted tasks never reached
training: 198 were not captured and 307 were captured but not examined by native
grading. The 6,144 task-update exposures that were examined were eligible and
reported trained; that statement covers only the selected subset.

In epoch 153, 451 of the 512 trained tasks had appeared in the previous twelve
training epochs; 61 were new to that window. Prior overlapping three-epoch
windows covered 1,205–1,245 distinct tasks from 1,536 task-update exposures.

## The quota also selects which tasks are practical to mine

Four correct and four incorrect responses are selected outcomes, not eight
unselected policy samples. Very easy and very hard tasks can require much more
search to complete that quota. In an illustrative independent 50-attempt search,
the probability of obtaining at least four of each outcome is about 0.16% for
a task success probability of 1% or 99%, 23.96% at 5% or 95%, and 74.97% at 10%
or 90%. These calculations ignore time limits, repeated outputs and truncation;
they are not estimated production success rates.

The current public CLI defaults to 50 attempts per search call and shuffles the
allowed task list. The 1,000-nonce protocol range is not a default 1,000-attempt
search. Custom miners may allocate their effort differently. Reordering the
manifest alone therefore does not steer the default miner; changing its allowed
indices does.

The observed omissions are not simply the hardest dataset labels. Level 1 is
7.56% of the allowed pool but 2.82% of task-update exposures; Level 5 is 30.52%
of the pool and 38.13% of exposures. Dataset levels are not calibrated difficulty
for the current model.

## The first change is a broad task cooldown

Starting with epoch 154, the manifest temporarily excludes 256 of the most
frequently trained tasks in the previous twelve finalized epochs, retaining
6,490 choices and the fixed calibration tasks. Standard miners read the new
allowed list automatically. Custom clients with fixed indices must also obey it.
The first three trial epochs, 154–156, were fixed before their outcomes were
available. See the [published mechanism](https://github.com/AffineFoundation/affine/blob/d59a6bde8bc88c975b3b2523732b2d2af9a857a0/docs/TASK_COVERAGE.md).

Filtering historical submissions using only each epoch's preceding history
gives this supply check:

| Tasks temporarily excluded | Minimum captured distinct tasks | Epochs retaining at least 512 |
| --- | ---: | ---: |
| 0 | 802 | 12/12 |
| 256 | 701 | 12/12 |
| 512 | 610 | 12/12 |
| 1,024 | 436 | 4/12 |

This measures retained historical supply, not how miners will adapt. Captured
tasks are not all independently verified. Capture concurrency increased at
epoch 152, so 152–153 are also reported separately in the frozen analysis.

The current trial stays at 256. A later 512-task trial is a candidate only if
novelty barely changes while task supply, participation and epoch duration remain
healthy. Concentrating everyone into a small shared pool could increase
same-epoch collisions under duplicate-zero rewards.

## What would count as useful progress

Measure novelty and breadth in actual training, alongside submitted supply,
subject/level mix, collisions, external participation and full epoch duration.
Broader coverage does not establish better learning. The separate independent
[step-60 evaluation](LIVE_STEP60_RESULTS.md) found a substantial regression, so
training-method and data-quality experiments continue alongside this trial.

No new model calls, historical rule changes or penalties were used for this
coverage analysis. Original signed epoch records and successful training-job
bindings were checked; the underlying trainer reports are operational evidence,
not cryptographic proofs of physical execution.

Evidence SHA256:

- Frozen reference census: `cf68059fe879ddf8cd596603c0e1d9bd4d97e704971546629dac12481088862e`
- Coverage and selection analysis: `211e4f215fc14ebcabaa91b13552cdcdb016f4eefb280044024a88557210e9ef`
- Task registry: `77a4524abc279d0e6e95ec87d0e5604f501c8e409ac5ab656ecabf060ab50fe3`

## First completed trial epoch: 154

Epoch 154 closed normally after 61 minutes 1 second, exceeding the one-hour
target by about one minute. Its original training job succeeded and published
checkpoint `9e374468d205803b6719de1f379cdac6b7b6d01a55e2a15b070b817eac5fb9ad`.
The next epoch opened automatically. No manual recovery was used.

| Metric | Epoch 152 | Epoch 153 | Epoch 154, cooling active |
| --- | ---: | ---: | ---: |
| External submitting identities | 212 | 238 | 251 |
| Committed batches | 1,817 | 1,889 | 2,138 |
| Distinct committed tasks | 1,455 | 1,514 | 1,640 |
| Distinct captured tasks | 1,304 | 1,209 | 1,227 |
| Distinct trained tasks | 512 | 512 | 512 |
| Trained tasks absent from preceding 12 training epochs | 62 | 61 | 70 |
| That novelty share | 12.1% | 11.9% | 13.7% |
| Training share on the fixed reference's top 256 tasks | 21.7% | 22.3% | 0.8% |

The first observation shows miners moving away from the most repeatedly trained
tasks without an observed shortage of eligible training data. The increase in
training novelty is modest, and most selected tasks still appeared in the recent
window. Among captured tasks, 182 were new to that window; 112 of them were not
selected for native grading or training. Their native eligibility was unknown
at this first review; the separate closed-epoch follow-up below checks it.
Giving such candidates priority is a separate trainer-side option that would
not require a miner update; it is not active in this trial.

The fraction of committed batches sharing a task with another external identity
rose from 36.9% to 43.5% between epochs 153 and 154, while submissions also grew.
This is task overlap, not identical rollout content or the authenticated reward
collision rate. It remains part of the tradeoff to monitor.

These are descriptive results from one of the three prespecified epochs, not
evidence of improved held-out performance or a causal effect of cooling. Epochs
155 and 156 also belong to the fixed trial. The analysis reads finalized records only.

First completed trial analysis SHA256:
`b4b48acc00cfca85d9830a45d8dfdf12b0aa7f673baea1ca5a4023cc3a94ea4b`.


## Coverage across the complete current run

The authenticated records for all 64 finalized updates, epochs 91–154, contain
26,913 task-update exposures but only **2,923 distinct tasks out of 6,746**
(43.3%). That leaves 3,823 tasks unused by training in this run. Model and
optimizer lineage is continuous across all 64 records; there are no missing
epochs in this census. This is training coverage, not a count of every task
miners attempted or proof that the remaining tasks are solvable under the quota.

“Absent from the previous twelve epochs” also includes tasks trained earlier.
The distinction matters:

| Metric | Epoch 152 | Epoch 153 | Epoch 154 |
| --- | ---: | ---: | ---: |
| Captured tasks absent from previous 12 training epochs | 145 | 147 | 182 |
| Captured tasks never previously trained in this run | 86 | 81 | 81 |
| Actually trained tasks new to this run | 37 | 33 | 34 |
| Captured, never-trained tasks not examined for native grading | 49 | 48 | 47 |

The first cooling epoch mostly brought back older tasks rather than increasing
the count of entirely new captured tasks. It also supplied 47 never-trained
tasks that the current training selection did not examine. The census alone
does not establish their eligibility; the follow-up below checks those examples.

A separate, inactive candidate prioritizes captured tasks absent from the
preceding twelve training epochs before repeated tasks, retaining native grading,
the same 512-task limit and the same time budget. All 182 such candidates in
epoch 154 fit below that limit, so this would put the 47 never-trained candidates
in the first selection tier without needing another priority rule or miner
upgrade. Invalid and unresolved examples would still be excluded. Historical
outputs remain unchanged, and the fixed 154–156 cooling trial remains in place
before any decision on a later activation.

This finding supports addressing trainer selection alongside miner incentives.
It does not establish that repetition caused the held-out regression or that
broader coverage will fix the training objective. The controlled learning-method
studies continue separately.

Additional evidence SHA256:

- All-run coverage: `1546c3f5fdc63100bfb307896117e80b5f1f1a2cce682e49ac65424803daca74`
- Recent versus all-run novelty: `7b22f54e2804ce975ddb0ab24f80fe74b855fbf1ead8545c0956a1620a9fbbb6`

## Second completed trial epoch and the skipped-data check

Epoch 155 finished in **56 minutes 43 seconds** and epoch 156 opened automatically.
There were 249 external submitting identities, 2,155 committed batches covering
1,596 tasks, and 1,201 captured tasks. Training used 512 tasks, including 73 absent
from the previous twelve epochs. Only 31 were entirely new to this run, even
though the captured data included 68 never-trained tasks. That leaves 37
never-trained candidates unexamined in this epoch. Their eligibility has not
been tested. Cumulative training coverage reached 2,954 of 6,746 tasks.

Separately, we checked all 112 previously unused, prior-window-new task batches
from **closed epoch 154**, preserving their original checkpoint, prompts,
tokenizer, inputs and grading rules. This was one fixed diagnostic, without
resampling or selecting a favorable subset. All 112 batches passed native
grading: 448 pairs and 896 responses matched their claimed outcomes. That
includes all 47 never-trained tasks and all 65 older returning tasks. No
indeterminate grades occurred. The filter took about 150 seconds on four CPU
cores; no model inference or optimizer update was performed.

Combining these results with epoch 154's original selected-batch grading gives
182 known eligible tasks absent from the previous twelve epochs, including 81
never previously trained in this run. A rule that prioritizes those tasks would
put all 182 ahead of recent repeats within the 512-task cap, leaving 330 slots.
The actual historical selection used 70 such tasks, including 34 never-trained
tasks. This is retrospective coverage arithmetic, not a replayed 512-task
training update or a prediction of the next epoch's supply and timing.

Passing native grading checks answer labels and the required batch composition.
It does **not** verify prescribed token sampling, honest historical execution or
the mathematical soundness of every reasoning step. This diagnostic changes no
historical rewards or model weights. It supports prioritizing underused eligible
data, while leaving the learning-method question open. The priority candidate
remains inactive pending the fixed trial and deployment review; existing miner
clients do not need an upgrade for that server-side selection change.

Additional evidence SHA256:

- Finalized 155 trial analysis: `356d25a7abfadee80c68eb7a3da27febdee8f3948855dcaecf74c06f46aab934`
- Finalized 155 all-run delta: `a99a28683387793ade533f1177ac93ff3ab1b257e294bad0c5764c983228236d`
- Original 112-task grading receipt: `9a1cb3878128b8e92ce9cb3d94ee0286502ae73b1e2700aa5bb8f5ed7bd1f83f`
- Original diagnostic capture: `085c73a41cee0548c1b2f90ec1c53cc2d29e94b2e7dae6dafa3d846922505e62`
- Coverage-only counterfactual: `8a12be0ab9edfd476e9c406f95f36f6a63b03fd936e6a54c3d8e0b662032116f`

## Completed 154–156 trial and the next selection test

Epoch 156 completed in **58 minutes 41 seconds**, with 251 external submitting
identities, 2,170 committed batches, 1,603 committed task indices and 1,184
captured task indices. Training used 512 tasks, including 66 absent from the
previous twelve training epochs and 28 never previously trained in this run.
Epochs 155 and 156 both finished within an hour; epoch 154 took 61 minutes.

Across the three trial epochs, distinct task counts were:

| Stage | Distinct tasks | Not trained during epochs 91–153 |
| --- | ---: | ---: |
| Committed | 2,400 | 197 |
| Captured | 2,173 | 163 |
| Native-checked and trained | 1,276 | 93 |

The 1,536 task-update exposures covered 1,276 distinct tasks. Cumulative coverage
through epoch 156 is **2,982 of 6,746 tasks (44.2%)**. Of the 3,764 still untrained
tasks, 3,660 had no commitment during this three-epoch trial. Only 70 were
captured but remained untrained; their eligibility is not established by these
counts. A different learner order can use neglected submitted data, but cannot
recover tasks absent from the submitted population.

Coverage is uneven by dataset label. Level 1 accounts for 7.6% of the mining
universe but 2.9% of distinct trained tasks in this trial; Level 5 accounts for
30.5% and 37.1%, respectively. Of all Level 1 tasks, 80% remain untrained in the
current run, compared with 46.5% of Level 5 tasks. Algebra and Prealgebra account
for 991 and 704 remaining untrained tasks. These labels do not measure difficulty
for the current model. The omitted population cannot simply be called the
hardest problems, and selected K4/L4 batches cannot estimate task success rates.

Recent-window novelty also overstates first-time coverage: the 209 trained tasks
absent from their preceding twelve epochs comprise 93 first-time tasks and 116
older returners. Report both quantities when judging the next change.

The scheduled epoch-158 selector checks captured tasks absent from the prior
twelve training epochs before recent repeats, with the same native eligibility
checks, 512-task cap and 600-second eligibility budget. It retains the broad
6,490-task pool, 256-task cooldown, K4/L4, nine-batch cap per UID and reward rules.
Existing miners do not need an upgrade. The fixed follow-up covers epochs
158–160, including any partial or fallback epoch rather than replacing it with a
more favorable result. Measure actual selection, first-time coverage, supply,
participation and time before deciding whether stronger rotation is warranted.

A separate optimizer CPU-thread improvement is scheduled at the same boundary.
Consequently this follow-up is descriptive operational evidence, not a controlled
estimate of either change's effect on model quality. The original released
step-60 regression predates the cooldown trial and cannot be attributed to it.
Controlled learning-method studies remain separate; broader coverage is not
evidence that held-out performance has improved.

Evidence SHA256:

- Fixed three-epoch aggregate: `a1d202c57e4ea6dc56fb7a65ffb1da4c5b55266fc2290009018ac03ad2a88eed`
- Original cached stage census: `36f95b4f6fa7204bacc6959e27568fc5aba9d06e22cccc3e6ad502f80029b2d7`
- Independently reproduced stage unions: `2734ff55474fae3649f381db20004c34f47b7781faa6e13c60f0e9b9b59f097f`
- Independent union review: `9869e5c486beb6b090bf00459e940c6e26ffa9c494e2cc0d607ce9324e2ec0ff`
- Subject/level coverage analysis: `99649dcb60d3826433ef3893bfb544dcb817981e292af1f8948b3984e573e80a`
- First-time versus older-returner decomposition: `5e02ec85481db0cc545aa33dcfdd4f39da036e04b0a965ece3249275b13bfc15`
