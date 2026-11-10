# Documented check — host restart during eval

**Claim (falsifiable):** After a host process restart, a PIT item whose box is
already past `train.done` and mid-detached-eval is re-adopted by reading
`markers/`, `job.pid`, and `RESULT.json`. The host does **not** re-run train,
does **not** kill/re-launch the live eval while the pid is alive, and does
**not** release the box.

**Automated proof:**

* `ops/parallel_eval/pit/test_box_job_restart.py::TestBoxJobRestart.test_restart_mid_eval_resumes_to_result_without_release`
  — box_job controller polls an in-flight eval pid to marker without relaunch.
* `ops/parallel_eval/pit/test_box_job_restart.py::TestBoxRunnerMidEvalReadopt.test_host_restart_mid_eval_skips_train_and_polls_pid`
  — BoxRunner `run()` with `train.done` present skips `train_v834e`, resumes
  eval via the same controller, reaches RESULT.

**Why it matters:** with `max_boxes=5` there is almost never a moment when
nothing is in eval, so every host install/restart must re-adopt cleanly.
