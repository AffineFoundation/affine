# SN120 probe grade-stall fix (2026-11-01; Jacob: condition for the next gate switch)

Incident (2026-11-01 13:31-14:06 CR): round r20261101T193112Z-32fa9510 head baseline block 0 (2000 tasks x k4): 8000 samples
generated, 7996 graded, then one grade-pool worker on the probe pod died: math_verify's SIGALRM
`TimeoutException` (a BaseException) escaped sympy inside the lenient grader (`except Exception` does not catch it).
multiprocessing.Pool silently replaced the worker and lost that task, so probe_worker waited forever with its pid alive; the gate
only salvages runs that are no longer "running"; the variance-levers watchdog auto-reverted after 35 min.

## Fix 1: probe_worker.py (probe pod /root/affine-env-probe; part of the round fingerprint -> installed only at a round boundary)
* `grade_job` wrapper in the pool worker: writes a start marker (/dev/shm/sn120-grade-<run pid>/<worker pid>), catches any
  BaseException from the grader -> `indeterminate` ("grade-crash ..."), disarms a leftover SIGALRM timer after every job.
* parent supervisor (every 2 s): a job whose worker pid is dead (2 passes) -> `indeterminate` "grade-worker-died"; a job running
  longer than `grade_timeout_s` (plan, default 300 s; sandbox envs `sandbox_grade_timeout_s` 1800 s) -> its worker is killed by
  exact PID, job -> `indeterminate` "grade-timeout". Either way the pool is rebuilt (new Pool, other in-flight jobs resubmitted
  under new job ids, old pool terminated + any survivors killed by exact PID after 15 s); every event is logged to worker.err and
  counted in progress.json / result.json `grade_supervisor`.
* Semantics: `indeterminate` is the existing outcome of any grader exception: the sample counts in n and not in c (not passed,
  i.e. graded as wrong for the pass rate), recorded in x / unresolved_reasons. Same code path for head baseline blocks and
  candidate runs, so head and candidates are graded identically. ("Excluded for both" would need cross-run coordination and is
  not how the probe treats any other grader failure.) Normal samples are graded exactly as before.

## Fix 2: gate_service.py (affinevalidator, RA/20261026-promotion-gate/code)
* `stall_salvage()`: a run that is still "running" with every sample generated (progress.generated == tasks x k) and its graded
  count (progress.samples) unchanged for `stall_salvage_s` (EPOCH-TIME.json, default 300 s; 0 = off) -> the existing
  `podctl finalize-run <run> 30`; logged as `heldout_baseline_stall_salvage` / `gate_run_stall_salvage` (no "error" in the event
  name). The partial then goes through the normal result path: baseline blocks need min_cover 0.95 of their tasks (else partial
  retry), candidate runs 0.9 (else rerun). Wired into head baseline blocks and candidate runs.
* The variance-levers anchors are untouched: `patch_levers.py revert` + `apply` round-trip byte-exact on the new file.

## Tests (offline; test/)
* `run_case.py` + stub env (`src/subnet/environments.py`: task 1 hangs, task 2 os._exit()s its worker, task 3 raises a
  BaseException TimeoutException) + `fake_vllm.py`. Original worker: hangs in all three cases (44/48 or 8/48 graded).
  New worker: completes every case, faulted tasks n=4 c=0 x=4 with reasons grade-timeout / grade-worker-died / grade-crash,
  all other tasks graded identically to a fault-free run; 400-task / 16-worker stress run identical on every clean task.
* `hung_finalize.py`: `podctl finalize-run` on a hung ORIGINAL worker writes the partial in ~3 s and the worker exits
  (basis for cleaning the stale pid 2977240).
* `test_gate_stall.py`: salvage fires only when all samples are generated and the graded count is unchanged >= 300 s, not twice,
  off at 0; candidate path returns the 1999/2000 partial.

## Install / guard / revert (stallfix.py)
`stallfix.py arm <epoch>` waits for a DECISION with no open held-out round (the round just closed), stops the gate (< 90 s),
installs both files (backups in backups/sf-<ts>/ and on the pod probe_worker.py.bak-pre-stallfix-<ts>), starts the gate, then
guards the first round on the new code (auto-revert on gate errors / inactive / baseline not done 25 min / fail-closed / epoch
stall; it closes that round so no round changes code mid-way). Manual revert: `WHY=... python3 stallfix.py revert`.

## Live (2026-11-01)
- Installed 15:59:50 CR right after the e491 GREEN closed round r20261101T203403Z-1ba515be (no open round; gate stopped ~7 s).
- First round on the new code r20261101T220024Z-394279d5 (opened 16:00:24 CR at e493; n 2000 / k_cand 8 / pooled sigma_run
  unchanged): head baseline 733.5 s (block 0: 2000 tasks / 8000 samples in 426 s, blocks 1-3: 235 kept tasks each), no
  supervisor events; e493 decided RED z -0.213 at 16:27 CR (k_cand 8, result->decision 346.5 s). Levers watchdog untouched.
- The stale idle worker of the stalled round (pid 2977240 + 161 pool children) was cleaned up with
  `podctl finalize-run heldout-32fa9510-H-b0-k4 30` (closed round, result never used).
- Revert: `WHY=... python3 stallfix.py revert` (restores backups/sf-20261101T215942Z + the pod probe_worker.py.bak-pre-stallfix-*),
  or `stall_salvage_s: 0` in EPOCH-TIME.json to switch only the gate salvage off.
