# Public SQL candidate controls

`subnet.public_sql_candidates` derives a pair of SQLite query proposals from the original public schema and question. It has no database, private reference query, grader, task-file or heldout-data input. The current grammar supports the sixteen original department-management mining tasks. This is a curated public solver; it is not unrestricted model generation or a general SQL solver.

`ops/probe_public_sql_candidates.py` checks those proposals with the isolated original private grader. The operator probe loads the complete public/private collections, but processes mining indices 0–15 only. The actual controls produced one positive and one negative proposal for each of those sixteen tasks. Heldout indices 16–31 were not graded by this research probe.

A prospective isolated harness resolves candidates from each current public question rather than reusing task zero's query. Its generator revision and exact bytes are pinned in the operator configuration. Existing original bash exploration steps remain intact. Tokenizer admission measured all sixteen proposed pairs within the 128-token output budget, with actual lengths from 15 to 109 tokens. Native positive/negative controls and token lengths do not establish reachable sampling probabilities: target-model class-distribution measurements and complete proof/replay admission remain necessary before deploying this harness.

The existing live SQL controller remains on its prior immutable source and narrow task-zero policy. These controls do not count as additional trained tasks, completed epochs, or evidence of heldout improvement.

The remote diversity qualification has completed with the approved 135M
checkpoint `d5347dc1c9f59da0ec574fba01314ff6c637a12126b441fd27f0bbd4af481d4b`.
Across the sixteen original mining tasks, 77 bounded candidate-policy attempts
retained fourteen positive and twelve negative rollouts. Indices
`0, 1, 2, 3, 4, 5, 7, 8, 9, 11` yielded a positive/negative pair. A separate
model reload verified all 26 retained rollouts with full probability checks,
strict TOPLOC and original native replay. This is the public query proposal
policy described above; it is not unrestricted autoregressive search.

Root independently authenticated the signed plan/receipt, checked all 67
pinned source modules and sixteen actual ZIP artifacts, and checked retained
probability arrays and proof framing. Root did not rerun these GPU model
computations locally. Evidence is in
`state/native-sql-common/diversity-model-control/root-artifact-evidence.json`.
These standalone controls qualify task admission; broader common-epoch
training and held-out improvement require their own completed evidence.
