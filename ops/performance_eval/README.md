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
(temperature 1.0, top_p 0.95, top_k 20, max 12,288 output tokens / 16,384 context for kings promoted after the 2026-11-02 12K switch, 8,192 / 12,288 before; run_king.sh 5th arg), same vLLM 0.32.0 serve flags,
same grader (final channel after `</think>`, exactly one balanced `\boxed`, math-verify 0.9.0 vs the reference answer,
then the probe's lenient fix allan-lenient-v1-786cdb9; capped replies are fails). 1 sample per task, fixed seed
`sn120-perf-v1`, seed(set, task, j) = int(sha256(P:set:task:j)[:12], 16). SE = task-clustered.

Informational only. Nothing in the gate, promotions, weights or the trainer reads it. The holdout is never used by
the gate (gate_core draws from range(7496) of the train-split env) and `holdout_guard` in performance.json checks every
miner free file against it (exact or Jaccard >= 0.9 = match).

Files: `perf_service.py` (host loop: kings from the gate ledger, fetch from R2 via the gate's presign, run on the pod,
publish), `perf_eval.py` + `run_king.sh` (pod), `fetch_king.py`, `holdout.py` (freezes the holdout), `build_tasks.py`,
`site/perf.js`, `site/perf.css`, `llms_perf.py`, `activate_perf.sh`, `revert_perf.sh`, systemd units.
Kings: every head since E0 (`head_history.py`: head of epoch e = the lifecycle TRAINING-RESULT input checkpoint; 172 heads, 171 still in R2,
`r2_kings.py` lists the bucket read-only) plus new GREENs from the gate ledger. Scheduling: the newest king first; idle time backfills
history in tiers (A: one king per 25 epochs, B: one per 10 epochs + every gate GREEN, C: the rest; oldest -> newest), a newer GREEN preempts a backfill.
The chart's x axis is the crowned epoch.
Pod disk keeps only the king being evaluated and the next one; everything is on plain disk (/data), never /root (gocryptfs).
Revert (validator host): `bash ~/subnet120-rewrite/state/root-audits/20261031-performance-eval/code/revert_perf.sh` (`POD=rm` also removes the pod).

## Second pod (history backfill, 2026-10-31 11:10 CR, Jacob)
- Pod 1 `calm-orbit-ef` (2xH200, primary): newest king first, then history; the only pod that publishes performance.json.
- Pod 2 `eager-wolf-98` (1xB200, lium id 2ad91129-cb87-4070-aa9e-942ee07ac329, unit `affine-perf-eval-2`, `PERF_POD=eager-wolf-98`): history only, never the newest king.
- No double scoring: each king is claimed with an O_EXCL file in `state/claims/<king>` holding the pod name; released on success/failure/preempt; a pod releases only its own stale claims on restart.
- Same seed, prompts, sampling and grader on both; pod 2 first re-scores c03ad139 as a cross-check (`state/xcheck/LATEST.json`, |diff| vs SE).
- Every performance.json row records `pod`, `pod_lium_id`, `gpu`.
- Auto-teardown: when pod 2 finds nothing left to claim, it runs `lium rm 2ad91129-cb87-4070-aa9e-942ee07ac329 -y` (exact id), logs `teardown` and appends to PROGRESS.md; pod 1 stays.
- `revert_perf.sh` stops both units and removes pod 2 by exact id if it still exists (pod 1 only with POD=rm).
