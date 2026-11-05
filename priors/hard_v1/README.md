# priors/hard_v1: the hard-v1 slice of the gate pool (gate-pool-v2)

hard-v1 = 2,269 harder math problems appended to the public gate pool as `dapo:11813..dapo:14081` (env `dapo`, field
`slice: "hard-v1"`), giving `miner/data/gate-pool-v2.jsonl` (21,574 tasks; v1 rows byte-for-byte unchanged). Live from the flag
change of 2026-11-05 13:43 CR (epoch 767): the first gate round opened on the next new head draws from v2.

Sources: CrystalMath-rewritten (ycchen/Crystal-Math-Preview @03fc7714; per-row licences) and INTELLECT-3-RL math
(PrimeIntellect/INTELLECT-3-RL @a9eb9183; no licence declared by the publisher).

| file | what |
|---|---|
| `build_hard.py` | normalise, gradeable-answer filter with the live grader, dedupe (vs gate-pool-v1 and all of DAPO-Math-17k, cross-source, within-source), decontaminate (AIME 2022-2026, AMC, MATH-500 / MATH test, GSM8K, arXivMath, OlympiadBench, Omni-MATH, OlymMATH) |
| `holdout_check.py` | drops candidates matching the private perf holdout (hash / normalised hash / token-3-gram Jaccard >= 0.9); 0 matched |
| `drive_32k.sh`, `orch.py` | difficulty filter: the live gate harness (T 1.0, top_p 0.95, top_k 20, thinking on, 32,768 output / 36,864 context, live grader) on king 8389b90b (E761), adaptive rollouts: 4 on every candidate, +4 if not 4/4 solved and < 3/4 capped, +4 more if 1-6 of 8 solved and < 4/8 capped |
| `analyze_32k.py` | keep rule: solve rate 0.05 <= p <= 0.6 and cap-hit < 0.33 over all of a task's rollouts |

Filter result (aggregates): 14,077 candidates, 14,075 sampled (2 never graded: grader timeout), 107,792 rollouts; dropped p < 0.05: 5,884,
p > 0.6: 5,395, cap-hit >= 0.33: 527; kept 2,269 (INTELLECT-3 2,001, Crystal 268), mean solve rate 0.32, pooled cap-hit 0.028.
Per-task solve rates are not published (they seed the gate's private IRT parameters). The pool itself is public; only each round's
draw is secret until it is revealed.
