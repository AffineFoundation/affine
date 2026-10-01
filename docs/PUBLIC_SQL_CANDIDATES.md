# Public SQL candidate controls

`subnet.public_sql_candidates` derives a pair of SQLite query proposals from the original public schema and question. It has no database, private reference query, grader, task-file or heldout-data input. The current grammar supports the sixteen original department-management mining tasks. This is a curated public solver; it is not unrestricted model generation or a general SQL solver.

`ops/probe_public_sql_candidates.py` checks those proposals with the isolated original private grader. The operator probe loads the complete public/private collections, but processes mining indices 0–15 only. The actual controls produced one positive and one negative proposal for each of those sixteen tasks. Heldout indices 16–31 were not graded by this research probe.

A prospective isolated harness resolves candidates from each current public question rather than reusing task zero's query. Its generator revision and exact bytes are pinned in the operator configuration. Existing original bash exploration steps remain intact. Tokenizer admission measured all sixteen proposed pairs within the 128-token output budget, with actual lengths from 15 to 109 tokens. Native positive/negative controls and token lengths do not establish reachable sampling probabilities: target-model class-distribution measurements and complete proof/replay admission remain necessary before deploying this harness.

The existing live SQL controller remains on its prior immutable source and narrow task-zero policy. These controls do not count as additional trained tasks, completed epochs, or evidence of heldout improvement.
