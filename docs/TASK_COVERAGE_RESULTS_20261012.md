# Closed-epoch task coverage: epochs 142–153

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

The first cooled epoch was training when this report was prepared. No coverage
benefit or held-out improvement from cooling is claimed yet.
