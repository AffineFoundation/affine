# PIT box pipeline: entry points for a controller

This document is the contract between the PIT box pipeline and a controller that
calls it (for example the `BoxAgent` of the batch controller). It gives, for each
entry point, the command or the call, the inputs and the output JSON.

Operator steps are in `docs/PIT.md`.

## Terms

| Term | Meaning |
|------|---------|
| box | A rented GPU machine. It holds no credential. |
| host | The validator machine. It holds the credentials and starts work on a box through ssh. |
| HEAD | The checkpoint that training starts from. |
| candidate | The checkpoint that training makes. Its id is a sha256 (64 hex characters). |
| bf16 | The served weights of a checkpoint (16 bit). |
| fp32 master | The 32 bit copy of the weights that the optimizer keeps. Its digest is a sha256. |
| job directory | One directory on the box for one item: `/workspace/pst/pit-jobs/<item id>`. |
| stage | One step of a job: `download`, `train` or `eval`. |
| marker | The file `markers/<stage>.done` in the job directory. It says that a stage ended, and how. |
| probe | The evaluation of a checkpoint: vLLM serves it, and a fixed list of tasks runs against it. |
| layout | How vLLM serves a probe on one box (GPUs for each server, vLLM version, limits). |
| output prefix | The start of each R2 object key that one run writes. |

## What "stable" means

- The names, the arguments, the required keys and the exit codes in this
  document do not change.
- A new key can come in an output JSON. A reader must ignore a key that it does
  not know.
- A change that is not compatible gets a new name or a new `version` value.

`ops/parallel_eval/pit/test_box_entry_points.py` holds these rules as a test.

## Box paths

These are constants of `ops/parallel_eval/live/pe_box.py`.

| Constant | Path on the box | Use |
|----------|-----------------|-----|
| `PST` | `/workspace/pst` | work directory of the trainer |
| | `/workspace/pst/in` | inputs of the trainer |
| | `/workspace/pst/out` | outputs of the trainer |
| | `/workspace/pst/train.log` | log of the trainer |
| `TW` | `/workspace/affine-mimo-trainer-prod` | trainer packages |
| `FP32` | `/workspace/affine-mimo-trainer-prod/fp32-master` | fp32 master store |
| `KDIR` | `/opt/plain/probe/affine-env-probe/ckpt` | checkpoints that the probe serves |
| `BOX_ROOT` | `/root/affine-env-probe` | probe directory (`podctl.py`, `runs/`) |

## Names that a controller loads

The batch controller loads these names from this code when it starts (its file
`batchctl/hostlink.py` has the list). A name that is not there stops its start.
Do not rename one, and do not change its arguments.

`P` is the ssh handle of one box: an object with
`run(cmd, inp=None, timeout=..., check=True)` that runs a shell command there.

Helpers that prepare a box. Call them in this order:

| Name | Call | What it does |
|------|------|--------------|
| `pe_box.boot_helpers` | `boot_helpers(P)` | First step on a new box. It makes the links and the directories, and puts `gate_fetch.py` and `bundle_fetch.py` in `/workspace/pst`. |
| `pe_box.push_probe_bundle` | `push_probe_bundle(P)` | It puts the probe bundle on the box (`podctl.py` and the vLLM environment). |
| `pe_box.bundle_trainer` | `bundle_trainer(P)` | It puts the trainer bundle and the trainer package `host-v834e` on the box, and checks each sha256. It returns a record. |
| `pe_box.ship_inputs` | `ship_inputs(P, epoch_dir=..., epoch=..., work_jd=...)` | It writes the trainer inputs to `/workspace/pst/in` and downloads the rollout documents (with retries, see "Fetch"). It refuses an unsigned dispatch. `work_jd` is a private scratch directory on the host. |
| `pe_box._ship_train_pit` | `_ship_train_pit(P)` | It writes `/workspace/pst/train_pit.py`. |

Other names:

| Name | Call | What it does |
|------|------|--------------|
| `pe_box._train_pit_inner_cmd` | `_train_pit_inner_cmd(expected_master=...)` | It gives the train command text. See "Train". |
| `pe_box.put` | `put(P, dest, data, mode='644')` | It writes one small file on the box (8 MiB at most). |
| `pe_box.put_tar` | `put_tar(P, local_dir, remote_dir)` | It copies a directory to the box. It does not empty the destination first. |
| `probe_serve.ship` | `ship(write_text, job_dir, plan, read_text=None)` | It copies the probe tool and the layouts to a directory on the box. It returns `{<file name>: <path on the box>}`. |
| `probe_serve.detached_command` | `detached_command(preamble=..., job_dir=..., candidate_id=..., plan_path=..., root='/root/affine-env-probe', box_args=())` | It gives the shell text of `probe_serve.py eval`. See "Probe". |
| `probe_layout.ProbePlan` | `plan.layouts`, `plan.primary`, `plan.eval_timeout_s()`, `plan.primary.concurrency()` | The serving layouts of a probe, the first one, the time limit of one probe in seconds, and the requests in flight for each server. |
| `pe_probe_plan.build_probe_plan` | see "Probe" | It makes the plan of a probe. The file `live/pe_probe_plan.py` can run on the box: it uses the standard library only. |
| `probe_fingerprint.of`, `.KEY`, `.FingerprintError` | see "Fingerprint" | The fingerprint of a probe. |
| `gpu_policy.effective_gpu_order` | `effective_gpu_order(order=None, allow_sharded_gpus=False)` | The GPU classes that a run can rent, in order. |
| `pit/box._load_public_harness` | `_load_public_harness()` | The public harness of the live lane: `harness` and `sampling_contract`. |
| `pit/pe_score.call_live_host_score` | `call_live_host_score(host_score, epoch=..., round_id=..., result=..., baseline_of=None)` | It runs the live decision function on one probe result. With `baseline_of`, the fingerprint check runs before the scorer. See "Fingerprint". |
| `live/pe_common.lium` | `lium(*words, timeout=180)` | It runs the Lium command-line tool. It returns `(exit code, stdout, stderr)`. Validator host only. |
| `live/pe_common.pod_of` | `pod_of({"id": ..., "ip": ..., "port": ..., "huid": ..., "name": ...})` | It gives `P` for one box. Validator host only. |
| `live/pe_score.score` | `score(epoch, rid, result, flag=None)` | The decision function of the live gate. Validator host only. |
| `live/pe_score.HELDOUT` | a path | The directory where the scorer reads its rounds. Validator host only. |

"Validator host only": the module does not load on a different machine.

The controller also imports these names. Keep them:

| Module | Names |
|--------|-------|
| `ops/parallel_eval/pit` | `ALLOWED_GPUS`, `SHARDED_GPUS`, `REQUIRED_GPU_COUNT`, `PRICE_CEILING_USD_H`, `ALLOW_SHARDED_GPUS_DEFAULT` |
| `pit/shadow` | `ShadowWriteError` |
| `pit/fakepass_sandbox` | `FakepassError`, `assert_paths_outside_live`, `live_roots_from_env` |
| `pit/queue_ingest` | `_epoch_index_of`, `_above_floor_or_pit_open`, `_epoch_dir_ready` |
| `ops/parallel_eval/durable` | `canonical` |

## Output prefix (R2)

One prefix says where a run writes in R2. Example:
`pit-shadow/batchctl/<run id>/`.

With the output prefix `P/`, the pipeline writes these keys only:

| Key | Content |
|-----|---------|
| `P/pending/<candidate id>/<file>` | fp32 master files, before APPLY |
| `P/pending/<candidate id>/model/<file>` | bf16 candidate files, before APPLY |
| `P/fp32-master/<candidate id>/<file>` | fp32 master files, after APPLY |
| `P/ckpts/<candidate id>/model/<file>` | bf16 candidate files, after APPLY |

APPLY is the step that makes a candidate the new HEAD. Shadow mode has no APPLY.

Rules for the prefix: each part between `/` starts with a letter or a digit, and
has letters, digits, `.`, `_` and `-` only. Any other prefix is refused
(`ArtifactUploadError`).

There are four ways to give the prefix. The first one that is set wins.

| Way | Scope |
|-----|-------|
| `pe_box.HostBoxIO(output_prefix=P)` or `pe_box.box_io_factory(P)` | the calls of that object |
| `with pe_artifact_upload.output_prefix(P):` | the calls in the block, in that thread |
| `run_host.py --r2-output-prefix P` (shadow mode only) | the process |
| variable `PIT_R2_OUTPUT_PREFIX=P` (read by shadow mode only) | the process |

Two box jobs of one process can use two prefixes: the first two ways hold for
one thread. A thread that starts inside the block does not get the prefix.

With no output prefix, the three variables `PIT_R2_PENDING_PREFIX`,
`PIT_R2_FP32_PREFIX` and `PIT_R2_CKPT_PREFIX` decide, as before.

### Shadow mode

Shadow mode is a run that observes and never changes live state.

- `PIT_SHADOW_STATE` must name the shadow state root: the one directory that the
  process can write. It must be an absolute path that is not a live directory,
  not in one and not above one. Without it the process does not start.
- The R2 prefix allowlist is `pit-shadow/`. An output prefix that is not under
  it stops the start (`ShadowConfigError`, exit code 2 of `run_host.py`).
- With no output prefix, shadow mode uses `pit-shadow/`.
- At the start, `shadow_guard.enter_shadow()` sets `PIT_STATE`, `PIT_UPLOAD_ROOT`
  and `PIT_FP32_PENDING_INDEX` to places in the shadow state root, and the R2
  variables to places under the output prefix.
- After the start, each R2 write, copy, delete and signed write URL outside the
  allowlist raises `ShadowWriteError`. This includes a prefix that a caller
  gives later as an argument.

## Train

### Box command: `train_pit.py`

```
<TW>/runtime-exact/venv/bin/python -B /workspace/pst/train_pit.py
```

It has no arguments. It runs one optimizer step.

Environment:

| Variable | Rule |
|----------|------|
| `AFFINE_FP32_EXPECTED_MASTER_HASH` | Necessary. The digest (64 hex) of the fp32 master of HEAD. The trainer stops if the master on the box has a different digest. |
| `AFFINE_FP32_BOOTSTRAP_OVERRIDE` | Must not be set. The trainer stops if it is set. |
| `PIT_ZERO3` | `1`: use the sharded trainer package `host-v834f` (the update runs over all GPUs). |
| `PIT_ALLOW_SHARDED_GPUS` | `1`: permit `PIT_ZERO3`. `PIT_ZERO3` without it stops the trainer. |
| `PIT_PER_GPU_MEM_GIB` | Memory of one GPU in GiB. Below 100, with `PIT_ALLOW_SHARDED_GPUS`, the sharded package runs. |

Inputs (the host or the controller puts them there before the start):

| Path | Content |
|------|---------|
| `/workspace/pst/in/manifest.json` | signed manifest of the training job |
| `/workspace/pst/in/dispatch.json` | signed dispatch. An unsigned one is refused. |
| `/workspace/pst/in/train_config.json` | trainer settings |
| `/workspace/pst/in/meta.json` | `{"optimizer_step": <int>, "epoch": <int>}` |
| `/workspace/pst/in/head.descriptor.json` | `{"id": <HEAD id>, "files": {<name>: <sha256>}}` |
| `/workspace/pst/in/actor/<HEAD id>/` | bf16 files of HEAD |
| `/workspace/pst/in/docs/<sha256>.json` | rollout documents of the dispatch |
| `<FP32>/<entry>/` | fp32 master of HEAD (`meta.json` and shards) |

The trainer checks its own package first (`host.checked(...)`, the ROOT
admission). A package that ROOT did not admit does not train.

Outputs:

| Path | Content |
|------|---------|
| `/workspace/pst/out/descriptor.json` | `{"id": <candidate id>, "files": {<name>: <sha256>}}` |
| `/workspace/pst/out/report.private.json` | trainer report (see below) |
| `/workspace/pst/out/**/model-00001-of-*.safetensors` | bf16 files of the candidate |
| `<FP32>/<entry>/` | fp32 master of the candidate |

Standard output: one JSON object on each line, with the key `event`. The
events, in order:

| Event | When | Main keys |
|-------|------|-----------|
| `root-admission-checked` | the package passed the ROOT check | `engine_fp32`, `host` |
| `zero3-policy` | sharded package only | policy of the sharded mode |
| `native-installed` | the trainer is installed | `dp_world` |
| `zero3-installed` | sharded mode only | `dp_world`, `dp_token_budget` |
| `actor-loaded` | HEAD is in memory | |
| `vram-preflight-ok` | sharded mode only | `certain_gib`, `estimate_gib`, `limit_gib`, `per_gpu_gib` |
| `optimizer-built` | | `step` |
| `signed-dispatch-prepared` | | `groups`, `rollouts`, `max_response_tokens`, `max_context_tokens` |
| `native-grpo-complete` | the update is done | `status`, `optimizer_step`, `trained_rollouts`, `trained_tokens`, `preparation_seconds`, `optimizer_seconds`, `preclip`, `loss` |
| `zero3-microbatch` | sharded mode only | token budget, steps, peak memory of each GPU |
| `staged` | the candidate files are on disk | `output_checkpoint` |
| `done` | success | `checkpoint`, `fp32` (`source`, `input_master_digest`, `output_master_digest`, `input_bf16`, `output_bf16`), `torch_max_alloc_gib`, `host` |

Success: the event `done` and exit code 0. Any other end is a failure.

Trainer report keys that a controller can read: `status`, `optimizer_step`,
`input_checkpoint`, `trained_rollouts`, `trained_tokens`, `preparation_seconds`,
`optimizer_seconds`, `preclip_gradient_norm` (the gradient norm before clipping),
`objective_loss`, `microbatch`, and
`optimizer.sgd.fp32_master` (`source`, `input_master_digest`,
`output_master_digest`, `output_store_entry`).

### The command text

`pe_box._train_pit_inner_cmd(expected_master=<digest>)` gives the shell text of
the train step. A controller uses this text and does not write its own.

The text does these things, in this order:

1. `rm -rf /workspace/pst/out`. A second start destroys the work of the first.
   Start one trainer only (see "Box job files").
2. It runs `train_pit.py` with `env -u AFFINE_FP32_BOOTSTRAP_OVERRIDE` and
   `AFFINE_FP32_EXPECTED_MASTER_HASH=<digest>`. Output goes to
   `/workspace/pst/train.log`.
3. It adds the line `rc=<exit code>` to `train.log`. `rc=0` is success.

`pe_box.detached_train_command(expected_master=<digest>, job_dir=<job directory>)`
gives the same text, and then writes `markers/train.done`:

| Key | Meaning |
|-----|---------|
| `ok` | `true` if `rc=0` is in the log and the event `done` names a checkpoint |
| `rc_ok` | `true` if `rc=0` is in the log |
| `checkpoint` | candidate id from the event `done` |
| `events` | the last event of each name |
| `candidate_id`, `candidate_sha256` | candidate id from `descriptor.json` |
| `candidate_path` | `box:<KDIR>/<candidate id>` |
| `box_candidate_dir` | `<KDIR>/<candidate id>` |
| `candidate_files` | `{<name>: <sha256>}` |
| `candidate_master_digest` | digest of the new fp32 master |
| `candidate_master_path` | `/workspace/pst/out/fp32-master-digest` |
| `training_report_summary` | `source_file_hashes`, `batching`, `tokens`, `update_size`, `gradient_norm`, `engine` |
| `report` | path of the trainer report on the box |

The keys after `events` are there only when the trainer wrote its outputs.
The command ends with exit code 0 if `ok` is `true`, else 1.

### Host call: `HostBoxIO.train_v834e`

```python
io = pe_box.HostBoxIO(output_prefix='pit-shadow/batchctl/<run id>/')
result = io.train_v834e(pod=pod, item_id=..., head_path=..., expected_master_digest=...,
                        batch_handle=..., epoch=..., epoch_dir=..., train_gpu=...)
```

It downloads the inputs to the box, trains, and uploads the two artifacts to
the pending prefix.

| Argument | Meaning |
|----------|---------|
| `pod` | Necessary. An object with `run(cmd, inp=None, timeout=..., check=True)` that runs a shell command on the box. |
| `expected_master_digest` | Necessary. Digest (64 hex) of the fp32 master of HEAD. |
| `head_path` | Necessary. `r2://head/<HEAD id>` (the answer of `download_head`), or the HEAD id. |
| `batch_handle` | Necessary. Path of the epoch directory, or of a file in it. |
| `item_id` | Name of the job directory. Default `pit-item`. |
| `epoch` | Epoch number. Default: the name of the epoch directory. |
| `epoch_dir` | Directory with `TRAINING-JOB.json` and `TRAIN-DISPATCH-FF.json`. |
| `train_gpu` | GPU class for the record. Default: read from the box. |
| `fp32_bootstrap_override`, `bootstrap_override` | Refused: `BoxIOError('bootstrap-override-refused-by-pit')`. |

Result keys:

| Key | Meaning |
|-----|---------|
| `ok` | `true` |
| `candidate_id`, `candidate_sha256`, `checkpoint` | candidate id |
| `candidate_files` | `{<name>: <sha256>}` |
| `candidate_path`, `box_candidate_dir` | where the bf16 files are on the box |
| `candidate_master_digest` | digest of the new fp32 master, as the host recorded it |
| `candidate_master_path`, `box_master_dir` | where the fp32 master is on the box |
| `fp32_master_source` | `disk` or `memory` |
| `used_fp32_bootstrap_override` | always `false` |
| `report` | path of the trainer report on the host |
| `report_doc`, `training_report` | the trainer report |
| `events` | train events |
| `train_gpu`, `pod_id`, `pod`, `box_work`, `venv`, `fetch` | records of the box |
| `info` | timings and records of each step. `info['train-start']` is `started` (this call started the trainer) or `adopted` (a trainer was already alive and this call waited for it). |
| `artifacts_uploaded`, `artifacts_pending` | `true` after the upload |
| `artifact_transport` | `uploaded-after-train-pending` |
| `early_upload` | record of the upload (`candidate_upload`, `master_publish`, `attempt`) |
| `live_apply_seed` | hint for APPLY. APPLY does its own check. |
| `defer_candidate_upload`, `candidate_on_box` | `false` after the upload |

Each failure raises `pe_box.BoxIOError`. Nothing is recorded as a success.

## Probe

### Box command: `probe_serve.py`

```
python3 probe_serve.py [options] <command> [arguments]
```

`probe_serve.py` and `probe_layout.py` must be in one directory on the box.
`probe_serve.TOOL_FILES` names them. The host function
`probe_serve.ship(write_text, job_dir, plan)` copies them there with the file
`probe-layouts.json`.

Options (before the command):

| Option | Default | Meaning |
|--------|---------|---------|
| `--root` | `/root/affine-env-probe` | probe directory on the box |
| `--python` | `python3` | interpreter that runs the pinned `podctl.py` |
| `--port` | `18555` | port of server 0 |
| `--poll-s` | `5` | seconds between looks at the servers |
| `--probe-poll-s` | `10` | seconds between looks at the probe run |
| `--settle-s` | `20` | wait after a stalled run is closed |
| `--drain-s` | `60` | longest wait for old servers to exit |
| `--gpu-wait-s` | `300` | longest wait for free GPUs |
| `--free-mib` | `2000` | a GPU with less memory in use than this is free |

Commands:

| Command | Arguments | Result |
|---------|-----------|--------|
| `start` | `--checkpoint <id> --layout <file> --venv <dir>` | Starts the servers of one layout and returns. Prints `{"gpu_groups": [[0, 1], ...]}`. |
| `stop` | none | Stops each server. Exit code 0. |
| `eval` | `--job-dir <dir> --candidate <id> --plan <file> --layouts <file>` | Serves and probes one item. Writes `markers/eval.done` and `RESULT.json` in the job directory. Exit code 0 on success, 1 on failure. A marker is written in each case. |
| `side` | `--checkpoint <id> --plan <file> --layout <file> --out <file> [--tag <name>]` | Serves one layout, runs one plan, stops, writes a side record for `probe_parity.py`. |

Input files:

- `--plan`: the JSON that `pe_probe_plan.build_probe_plan(cid, indices,
  plan_seed, k=8, max_tokens=65536, env_id='affine_gatepool', run_name=None,
  max_num_seqs=512, harness=None)` returns. Do not write the plan by hand.
- `--layouts`: `{"version": "pit-probe-layout-v1", "layouts": [<layout>, ...]}`.
  The first layout serves. The next ones are tried in order if it cannot start.
- `--layout`: a JSON file with a preset name (`"live"`, `"rtx5090-tp2"`,
  `"rtx5090-1gpu"`) or the settings of one layout.

The probe has no FP8: a server with an argument outside
`probe_serve.ALLOWED_FLAGS` fails the proof of the layout.

### Output: `RESULT.json` and `markers/eval.done`

The two files have the same content.

| Key | Meaning |
|-----|---------|
| `ok` | `true` only if `status` is `ok` |
| `status` | `ok`: the probe ended. `finalized`: the probe was closed early (stall or time limit) and its result is partial. `failed`: no probe result. |
| `candidate_id` | the checkpoint that was probed |
| `at` | time of the record (seconds) |
| `run_name` | name of the probe run |
| `probe_result` | the result of the probe worker, with `serve_layout` added |
| `error` | only with `failed`: class and text of the error |

A partial probe never gives GREEN.

`probe_result.serve_layout` says what served the probe:

| Key | Meaning |
|-----|---------|
| `layout` | the settings of the layout that served |
| `launcher` | how the servers started |
| `replicas` | one record for each server: `replica`, `pid`, `gpus`, `port`, `vllm_version` |
| `gpu_groups` | GPU ids of each server |
| `gpus` | names of the GPUs of the box |
| `runtime` | the Python environment of vLLM (`vllm_version` and more) |
| `serve_s` | seconds until each server answered |
| `concurrency_per_replica` | requests that the client keeps in flight for each server |
| `failed_layouts` | layouts that did not start, with the error |
| `queue` | how the requests went to the servers |
| `sampling` | `harness` and `sampling_contract` of the probe bundle, or `null` |

### Fingerprint

`probe_fingerprint.of(probe_result)` gives the fingerprint of a probe: one
record with the GPU class and count, the GPUs for each server, the number of
servers, the vLLM version, the serving limits and the sampling config. It reads
`serve_layout`, or the stamp under the key `probe_fingerprint`.

Rule: a candidate gets a decision only against a HEAD baseline with an equal
fingerprint. A difference, or no fingerprint on one side, is `no_decision`. The
scorer does not run.

### Host tool: `probe_parity.py`

It runs one checkpoint (HEAD) under two layouts with one plan, and compares the
pass rate of each task. Side A is the reference. A delta is B minus A.

```
python -m ops.parallel_eval.pit.probe_parity plan --checkpoint <id> --seed <text> (--indices <file> | --tasks <n> --pool-size <n>) --out <file>
python -m ops.parallel_eval.pit.probe_parity run-side --pod <file> --plan <file> --layout <name or file> --out <file>
python -m ops.parallel_eval.pit.probe_parity compare <side A> <side B> [options]
```

`plan` writes the probe plan that the two sides run. `run-side` runs one layout
on one box and writes a side record. `compare` reads two sides. A side is that
record, or a probe result (`probe_result` of `RESULT.json`).

Options of `plan`:

| Option | Meaning |
|--------|---------|
| `--checkpoint <id>` | the checkpoint (HEAD) that the two sides serve |
| `--seed <text>` | the plan seed. Each request seed comes from it. |
| `--indices <file>` | JSON file with the list of task indices |
| `--tasks <n>`, `--pool-size <n>` | without `--indices`: draw `n` tasks with an index below the pool size |
| `--k <n>` | rollouts for each task. Default 8. |
| `--max-tokens <n>` | the largest output of one rollout, in tokens. Default 65536. |
| `--env <id>` | environment id on the box. Default `affine_gatepool`. |
| `--harness <file>` | the public harness file. Default: the harness of this host. |
| `--out <file>` | where the plan goes |

Options of `run-side`. The checkpoint must be on the box.

| Option | Meaning |
|--------|---------|
| `--pod <file>` | JSON file with the handle of the box (the rent record) |
| `--plan <file>` | the plan that `plan` wrote |
| `--layout <name or file>` | a preset name, or a JSON file with layout settings |
| `--tag <name>` | name of this side. Default: the layout name. |
| `--work-dir <dir>` | directory on the box. Default `/workspace/pst/probe-parity`. |
| `--out <file>` | where the side record goes on the host |

Options of `compare`:

| Option | Meaning |
|--------|---------|
| `--max-shift <fraction>` | the largest change of the mean pass rate that is still parity |
| `--z-crit <number>` | a test beyond this value says that the layouts differ. Default 3. |
| `--top <n>` | tasks in the text report. Default 20. |
| `--out <file>` | write the full report as JSON |
| `--train-a <file>`, `--train-b <file>` | the trainer report of each side, for the preclip gradient norm |

Exit code of `compare`: 0 for the verdicts `parity` and
`no_difference_detected`, 1 for `different`, 2 for `inconclusive`,
`not_comparable` and an error.

The preclip gradient norm is the size of the gradient of one trainer update,
before the trainer limits it to `max_gradient_norm`. The report tracks it for
the two sides. It is not a test: it does not change the verdict or the exit
code.

`--train-a` and `--train-b` take the trainer report of the update that the
trainer of that side made (`report.private.json`), or a document that holds
it: `TRAINING-RESULT.json` of a live epoch, the result of `train_v834e`, the
training report on the host, or `markers/train.done`. The two updates must be
of one batch. A document without a gradient norm is an error (exit code 2).

The key `preclip` of the report:

| Key | Meaning |
|-----|---------|
| `in_verdict` | always `false` |
| `a`, `b` | the update of each side, or `null` when the side has no trainer report |
| `ratio_b_over_a` | norm of B divided by norm of A |
| `relative_difference` | the difference of the two norms, divided by the larger norm |
| `shadow_tolerance` | the largest `relative_difference` that the shadow pass criterion permits (0.05) |
| `within_shadow_tolerance` | `true` if `relative_difference` is not above `shadow_tolerance` |
| `same_job` | `true`: the two updates start from one checkpoint, with one batch, at one optimizer step. `false`: one of these differs; `ratio_b_over_a`, `relative_difference` and `within_shadow_tolerance` are `null` then. `null`: the documents do not say. |
| `differences` | names of the inputs that differ between the two updates (settings, turn parameters and fp32 master too) |

Keys of `a` and `b`: `preclip_gradient_norm`, `max_gradient_norm`,
`above_limit` (`true` if the norm is above `max_gradient_norm`), `engine`,
`ranks` (processes of a sharded update), `status`, `objective_loss`,
`trained_rollouts`, `masked_rollouts`, `trained_tokens`, `loss_tokens`,
`derailed_rollouts`, `input_checkpoint`, `data_sha256`,
`optimizer_step_before`, and `from_this_checkpoint` (`true` if the update
starts from the checkpoint of the two sides).

`ops/parallel_eval/pit/compare_train_reports.py` compares the same two trainer
reports rollout by rollout.

## Upload

### Box command: `pe_r2_box_upload.py`

```
python3 pe_r2_box_upload.py stat REQUEST.json
python3 pe_r2_box_upload.py upload JOB.json
```

The tool uses the standard library only. The last line of its output is one
JSON object. Exit code 0 if `ok` is `true`, else 1. Wrong arguments: exit code 2.

`stat` request: `{"trees": {<label>: {"dir": <directory>, "names": [<file>, ...] or null}}}`.
`null` means each file below the directory.

`stat` answer: `{"version": "pit-r2-upload-v1", "ok": <bool>, "trees": {<label>: {<file>: <bytes>}}, "missing": [...]}`.

`upload` job (the host makes it with `pe_r2_multipart.plan_upload`):

| Key | Meaning |
|-----|---------|
| `version` | `pit-r2-upload-v1` |
| `workers` | parts in flight at one time |
| `attempts` | tries for one part |
| `timeout_s` | seconds that one request waits |
| `files` | list of files |
| `files[].id` | `<label>/<file>` |
| `files[].path` | path on the box |
| `files[].bytes` | size |
| `files[].multipart` | `true`: more than one part |
| `files[].part_bytes` | size of one part |
| `files[].parts` | `[{"n": <part number>, "url": <signed URL>}]` |
| `files[].headers` | headers of a one-part upload |
| `files[].have_etag` | ETag of an object that R2 already has with this key and size |

The tool removes the job file: it holds signed URLs.

`upload` answer (the manifest):

| Key | Meaning |
|-----|---------|
| `version`, `ok` | as above |
| `files` | `{<id>: {sha256, bytes, multipart, part_bytes, hashed, sent, etags, reused}}` |
| `errors` | up to 20 texts. The causes come first. |
| `seconds`, `bytes_sent`, `bytes_reused` | totals |

`hashed` is the MD5 of each part as the hash pass read it. `sent` is the MD5 of
the bytes that went to R2. `etags` is the ETag that R2 returned for each part.

### Host calls

| Call | Use |
|------|-----|
| `pe_artifact_upload.upload_trees_from_box(P, candidate_id=..., put=pe_box.put, trees={<label>: {"dir": ..., "prefix": ..., "names": ...}})` | One box job sends file trees to explicit R2 prefixes. The host checks each object by size and ETag. |
| `pe_box._early_upload_artifacts(P, receipt)` | Sends the fp32 master and the bf16 candidate to the pending prefix. `train_v834e` calls it. |
| `HostBoxIO.upload_candidate(candidate_id=..., candidate_path='box:<dir>', pod=..., files={<name>: <sha256>})` | Sends the bf16 candidate to the pending prefix. |

`upload_trees_from_box` returns
`{<label>: {<file>: {sha256, bytes, r2_key, etag, part_bytes, multipart, parts, verify, reused}}}`.
It raises `ArtifactUploadError` on each difference, and records nothing then.

Records on the host:

| File | Keys |
|------|------|
| `<PIT_UPLOAD_ROOT>/<candidate id>/UPLOAD.json` | `id`, `sha256`, `files`, `verified_files`, `uploaded_at`, `source`, `transport`, `pending`, `objects` (`{<file>: {r2_key, bytes, verify, reused, etag, part_bytes, multipart, parts}}`) |
| `<PIT_UPLOAD_ROOT>/<candidate id>/CHECKPOINT.json` | `id`, `files` |
| `<PIT_FP32_PENDING_INDEX>/<candidate id>.json` | `files` (`{<file>: {sha256, bytes, r2_key, etag, multipart, parts, part_bytes}}`), `meta_json`, `master_digest`, `bf16_sha256`, `pending` |

## Fetch

### Box command: `gate_fetch.py`

```
python3 gate_fetch.py DESCRIPTOR.json DEST_ROOT URLS.private.json [--threads N] [--pidfile PATH]
```

It downloads the files of one checkpoint to `DEST_ROOT/<id>/`, checks each
sha256 against the descriptor, and writes `DEST_ROOT/<id>/.verified`. It removes
the URL file in each case.

| Input | Content |
|-------|---------|
| `DESCRIPTOR.json` | `{"id": <checkpoint id>, "files": {<name>: <sha256>}}` |
| `URLS.private.json` | `{<name>: <signed GET URL>}` |

Each request is tried again after a network error, with a pause that doubles.

| Variable | Default | Meaning |
|----------|---------|---------|
| `GATE_FETCH_ATTEMPTS` | 10 | tries for one request |
| `GATE_FETCH_BACKOFF_BASE` | 2 | first pause in seconds |
| `GATE_FETCH_BACKOFF_MAX` | 60 | longest pause in seconds |

The last line of standard output is one JSON record. It has no URL in it.

| `status` | Exit code | Other keys |
|----------|-----------|------------|
| `present` | 0 | `ckpt` |
| `materialized` | 0 | `ckpt`, `seconds`, `download_s`, `sources`, `threads` |
| `failed` | 1 | `ckpt`, `error`, `errno`, `errno_name`, `attempts`, `retryable` |

`retryable` is `true` when a new fetch can pass (a network error, not a
refused request).

### Host call: `pe_box.fetch`

`pe_box.fetch(P, cid, files, urls, dest, threads=32)` runs `gate_fetch.py` on the
box and returns `(seconds, bytes)`.

On a failure, the box command prints the line `PIT_FETCH_FAILED rc=<n>` and the
end of the fetch log on standard output. The host raises `BoxIOError` with the
real error of the box in it, for example
`box fetch <id> failed (rc=1): ... (errno ENETUNREACH) after 10 tries; a new fetch can pass`.

## Box job files

A job directory holds these files. `ops/parallel_eval/pit/box_job.py` names them.

| File | Content |
|------|---------|
| `markers/<stage>.done` | JSON marker of a stage that ended |
| `RESULT.json` | final result of the job |
| `job.pid` | process id of the detached stage |
| `job.stage` | stage of the process in `job.pid` |
| `job.log.<n>`, `job.log` | log of start number `n`, and a link to the newest |
| `start.claim` | `<owner> <stage> <time>`: the starter that has the start now |
| `start.flock` | lock file for the two files above |

A stage that has a marker does not run again.

The start of a detached stage is idempotent: two starters of one job give one
process.

- `RemoteBoxJobStore.launch_detached(job_dir, stage=..., command=...)` prints
  `STARTED <pid>` when it started the process. It prints `BUSY <pid>` when a
  process of that stage is alive; it starts nothing then.
- `BoxJobController.run_stage_once(stage, detached_command=...)` takes the
  start claim first. A starter without the claim waits and adopts the job that
  is alive. `BoxJobController.starts` records `(stage, 'started')` or
  `(stage, 'adopted')`.
- A claim ends 300 seconds after its last renewal. A starter that died holds
  the start for no longer.
