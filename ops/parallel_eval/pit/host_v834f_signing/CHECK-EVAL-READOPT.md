# Documented check — host restart during eval

**Claim (falsifiable):** After a host process restart, a PIT item whose box is
already past `train.done` and mid-detached-eval is re-adopted by reading
`markers/`, `job.pid`, and `RESULT.json`. The host does **not** re-run train,
does **not** call `place_cand` / `free_gpus` (which would TERM/KILL the live
eval process group), does **not** re-launch eval while the pid is alive, and
does **not** release the box.

**Real `train.done` shape:** live boxes write only
`{checkpoint, events, ok, rc_ok}` (plus enrichment when the writer can place).
Re-adopt must work with that sparse marker when `job.pid` is alive — it must
not require `candidate_path` / `candidate_master_path` to avoid falling through
to `place_and_pull`.

**Automated proof:**

* `ops/parallel_eval/pit/test_box_job_restart.py::TestBoxRunnerMidEvalReadopt.test_host_restart_mid_eval_skips_train_and_polls_pid`
  — BoxRunner with **sparse** train.done; asserts train/place/free not called
  and eval pid never signalled.
* `ops/parallel_eval/pit/test_mid_eval_sparse_readopt.py::TestHostBoxIOSparseMidEval.test_sparse_train_done_with_live_eval_never_place_or_free`
  — HostBoxIO.train_v834e mid-eval path; `place_cand` / `free_gpus` /
  `place_and_pull` raise if called.
* `ops/parallel_eval/pit/test_box_job_restart.py::TestBoxJobRestart.test_restart_mid_eval_resumes_to_result_without_release`
  — box_job controller polls an in-flight eval pid to marker without relaunch.

**Why it matters:** with `max_boxes=5` there is almost never a moment when
nothing is in eval, so every host install/restart must re-adopt cleanly.

**Early upload (e1073 / Fix B):** after first-pass `train.done` (eval not
live), HostBoxIO uploads verified bf16 + fp32 master and stamps
`live_apply_seed` so APPLY survives a dead box. Mid-eval re-adopt never
triggers that upload path's `free_gpus`/`place_cand`. Test:
`test_green_upload_before_apply.py` (box unreachable after GREEN; apply from
upload).
