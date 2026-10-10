# Committed unaudited learner

`committed-unaudited-training-v1` separates training intake from continuous
inference audits. The signed epoch manifest and approved source archive determine
which policies are active. Publishing code does not change an issued epoch.
Historical verifier-receipt policies keep their original audit gate.

## Current intake

A `small-commitment-pairs-v2` miner commitment binds each proof artifact and a
separate canonical training document of at most 2 MB. The collector authenticates
that commitment, freezes the document and checks identity, task/checkpoint
bindings, token and context limits, completed-answer framing, declared quotas,
allowed mining indices and held-out exclusions. The current batch has four
correct and four incorrect completed rollouts. A token-limit stop alone is not
an incorrect answer; missing or unfinished final answers do not fill a quota.

Before training, the CPU native selector independently grades the original
answers. Every required pair in an admitted batch must agree with the native
grader. Grading does not establish model execution or sampling consistency:
these inputs remain explicitly **unaudited**. Infrastructure errors are
indeterminate, not miner fraud. The trainer authenticates the original miner
signature and document digest, operator admission and selected population. It
also checks the task hash, seed and exact original prompt against the pinned
snapshot, tokenizer and template.

The currently activated capacity is up to nine batches per UID and 512 distinct
training tasks per epoch. Earlier signed manifests retain their old limits. The
pre-representative policy, including epoch 115, selected training inputs from
singleton task indices; multiple miners submitting the same task removed it from
that training inventory and gave those submissions zero contribution points.
Those historical rules remain unchanged.

## Distinct-task representatives

Signed epoch `nonpayable-live-reward-math-v1--1791622696-116`, starting
2026-10-10 at 09:02:08 UTC, activated `training_representative_policy` with version
`first-native-valid-distinct-task-v2-bounded-waves`. Miners continue using the same submission
format, sampler, nonce range, four-correct/four-incorrect quotas and public source.
The added field governs the learner only.

The collector preserves two inventories: the existing reward-eligible singleton
inventory and a separate authenticated pool of structurally eligible training
candidates. After freezing that pool, it journals a random draw that fixes the
order of tasks and candidate miners within each task. The native selector tries
candidates in that order and takes at most one fully native-valid batch per task.
If a candidate fails native eligibility, the next candidate for the same task can
be tried. Training still contains each task at most once, even when many miners
submitted it.

The initial activated policy admits at most 2,304 candidate documents and
4,608,000,000 input bytes, grades at most 256 documents per wave and spends at most
600 seconds on native selection. The separate signed training-task capacity caps
the result at 512 tasks. That is a capacity limit, not a claim that 512 tasks
were selected or trained; the original epoch reports supply the actual counts.
At the deadline, only completed native-valid batches
enter training. Unchecked candidates are neither accepted nor declared invalid.
A retry preserves the same draw, authenticated input bytes and completed waves;
it cannot reroll selection. An empty accepted result closes without an update and
preserves the parent checkpoint and optimizer counter.

This does **not** change duplicate-zero rewards. A colliding task can supply one
training representative while every colliding submission still receives zero
contribution points. Continuous audits already draw from the eligible committed
proof population, including collisions; reward eligibility is a separate filter.
Training selection does not turn unaudited inputs into proof-verified samples or
create new reward credit.

## Training and storage

The persistent FP32 master/Adam state, task-normalized objective, effective
learning rate and optimizer lineage remain unchanged by representative intake.
A separately signed intake-equivalence authorization can bind the narrowly
reviewed source change to original GPU qualification: all non-intake scientific
modules must match byte for byte, and installed native-child and identical-input
emission checks must pass. This explicitly records that the new source bundle
has not itself executed a fresh GPU qualification. It cannot authorize an
optimizer, objective or unrelated source change. Other research objectives need
their own qualification and release.

For the current `trainer-local-only-v1` lifecycle, master weights, optimizer
moments and counters remain in retained trainer-local storage, including managed
shared-memory state. The model checkpoint
and reports are published to R2; a signed state descriptor and acknowledged
same-job evidence bind that local state to the published checkpoint. The ordinary
path does not upload the full optimizer tensors every epoch. It never resets
Adam merely because an observation or publication retry occurs.

Local rollout downloads are disposable after completed, hash-checked durable
records and full-readback acknowledgments. Active model/state dependencies remain
until their working lifetime ends. Native child processes and descendants hold
the selection lease until they terminate, so a timeout cannot leave a late grader
writing into a successor selection.

The next epoch follows successful checkpoint publication and original-job state
adoption; it does not wait for all expensive audits. Independent evaluation
tracks held-out performance. More accepted tasks increase training coverage, but
do not by themselves establish learning improvement.
