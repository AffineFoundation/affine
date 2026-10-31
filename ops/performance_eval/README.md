# SN120 performance eval (Jacob, 2026-10-31)

Pass rate of every king (head) on two **fixed** sets, scored on its own GPU pod, published at
https://affine.io/performance.json and shown as the first tile ("Performance") on https://affine.io/.

Why: the held-out gate scores ~113-199 "kept" tasks per round, picked because the head solves them only sometimes,
from a 1,000-task draw of the pool. That sample is biased and changes every round, so gate pass rates are not
comparable across kings. These sets do not change.

| set | tasks | what |
|---|---|---|
| Full | 7,496 | the whole public affine_math pool (MATH-lighteval train rows with a `\boxed` answer, env index order) |
| Holdout | 4,976 | MATH test (5,000) minus 3 exact and 21 near (token-3-gram Jaccard >= 0.9, or >= 0.8 with the same answer) pool matches; list private, sha256 `08d994bab2da775b807b23ebaf098c35db95d14a12af7ced2a46d6df4ab6c734` |

Method (comparable with the gate probe): same prompt (system mandate + problem, MiMo chat template with thinking; the
full-set prompt token ids are the probe's own prep-cache ids, and our rendering reproduces all 7,496), same sampling
(temperature 1.0, top_p 0.95, top_k 20, max 8,192 output tokens / 12,288 context), same vLLM 0.32.0 serve flags,
same grader (final channel after `</think>`, exactly one balanced `\boxed`, math-verify 0.9.0 vs the reference answer,
then the probe's lenient fix allan-lenient-v1-786cdb9; capped replies are fails). 1 sample per task, fixed seed
`sn120-perf-v1`, seed(set, task, j) = int(sha256(P:set:task:j)[:12], 16). SE = task-clustered.

Informational only. Nothing in the gate, promotions, weights or the trainer reads it. The holdout is never used by
the gate (gate_core draws from range(7496) of the train-split env) and `holdout_guard` in performance.json checks every
miner free file against it (exact or Jaccard >= 0.9 = match).

Files: `perf_service.py` (host loop: kings from the gate ledger, fetch from R2 via the gate's presign, run on the pod,
publish), `perf_eval.py` + `run_king.sh` (pod), `fetch_king.py`, `holdout.py` (freezes the holdout), `build_tasks.py`,
`site/perf.js`, `site/perf.css`, `llms_perf.py`, `activate_perf.sh`, `revert_perf.sh`, systemd units.
Scheduling: the newest king first; idle time backfills older kings newest -> oldest; a newer GREEN preempts a backfill.
Pod disk keeps only the king being evaluated and the next one; everything is on plain disk (/data), never /root (gocryptfs).
Revert (validator host): `bash ~/subnet120-rewrite/state/root-audits/20261031-performance-eval/code/revert_perf.sh` (`POD=rm` also removes the pod).
