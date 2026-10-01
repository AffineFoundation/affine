# Public SQL candidate controls

`subnet.public_sql_candidates` derives a pair of SQLite query proposals from the original public schema and question. It has no database, private reference query, grader, task-file or heldout-data input. The current grammar supports the sixteen original department-management mining tasks. This is a curated public solver; it is not unrestricted model generation or a general SQL solver.

`ops/probe_public_sql_candidates.py` checks those proposals with the isolated original private grader. The operator probe loads the complete public/private collections, but processes mining indices 0–15 only. The actual controls produced one positive and one negative proposal for each of those sixteen tasks. Heldout indices 16–31 were not graded by this research probe.

A prospective isolated harness resolves candidates from each current public question rather than reusing task zero's query. Its generator revision and exact bytes are pinned in the operator configuration. Existing original bash exploration steps remain intact. Tokenizer admission measured all sixteen proposed pairs within the 128-token output budget, with actual lengths from 15 to 109 tokens. Native positive/negative controls and token lengths do not establish reachable sampling probabilities: target-model class-distribution measurements and complete proof/replay admission remain necessary before deploying this harness.

The SQL controller now uses the new immutable diversity source `c454ddb7f22fadc7`, rotating the ten original mining tasks with measured positive/negative admission. The first diversity epoch has now completed audit, training and paired evaluation on original task 9. Original heldout indices 16–31 remain fixed under this new source baseline. Standalone admission controls do not count as additional trained tasks, completed epochs or heldout improvement.

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

The first completed diversity epoch is
`nonpayable-native-sql-common-v3-diversity-1790855147-8`. Its actual frozen
three-turn positive/negative pair was independently audited, credited one
point, and consumed by one full-model update to checkpoint
`a01efb6390050dd9705d05eab905ce9dac2442966e2bb8c8372260a333510816`.
Root independently streamed all six published checkpoint objects
(272,585,280 bytes) and replayed both trajectories through the original local
actor/grader using the exact frozen diversity adapter and harness. All
observations and terminal outcomes matched. The sixteen fixed original
held-out tasks scored 0 before and after; this establishes broader task
training, not performance improvement. Root also checked the corresponding
public dashboard records and UID 131 grid value.

Private evidence: `state/native-sql-common/root-continuous-independent-evidence.json`,
`nonpayable-native-sql-common-v3-diversity-1790855147-8-root-checkpoint-publication.json`,
`nonpayable-native-sql-common-v3-diversity-1790855147-8-root-fresh-native-replay.json`,
and `state/dashboard/common-sql-seven-epoch-public-check.json`.
