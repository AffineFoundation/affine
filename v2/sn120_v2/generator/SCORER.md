# Generator frozen-king scorer

Build item 5: the pod-local scorer for SN120 mechanism 1. It uses the protocol
PR's `constants`, `decisions`, `pool`, `protocol` and `epoch` modules. Prompts
are the unmodified `sn120_v2.envs.math_pool.Task.messages()` messages. Grades
come from the production `GradePool` / `math_grader` (math-verify 0.9.0 plus
the v2 lenient rules), not a scorer-specific rubric.

The pod holds no R2 credentials and has no chain, site or publication client.
The only HTTP requests are literal loopback vLLM health/model/chat requests;
redirects, proxies, non-loopback hosts and HTTPS remote inference are refused.
The private seed is not logged or included in progress files. It is revealed
only in the final `out/results.json`.

## Start the frozen model

On the 8xB300 pod, with the folded BF16 base at `/root/dl/base`:

```bash
cd /root/gen/src/sn120_v2/generator
MODEL=/root/dl/base VENV=/root/venv ROOT=/root/gen bash serve.sh
```

This starts exactly eight independent vLLM OpenAI servers on
`127.0.0.1:8100` through `127.0.0.1:8107`, one per GPU. The launcher uses
`served-model-name null`, `bfloat16`, automatic (non-FP8) KV dtype,
`max-model-len 69632`, prefix caching, `gpu-memory-utilization 0.90`, and no
LoRA flags. Logs and PID files are under `/root/gen/logs` and
`/root/gen/pids`. `bash serve.sh stop` stops only matching scorer processes.

The scorer does not launch or replace the model itself. It checks all eight
health/model endpoints and requires the `null` served model before inference.
The supplied `head_version` binds the job and its evidence; the operator must
serve the corresponding frozen base before starting the scorer.

## Private job

The control plane writes `/root/gen/jobs/<epoch>/job.json`. Its strict fields
match `snapshot.scorer_job()`:

```json
{
  "version": "affine-generator-scoring-job-v1",
  "epoch": "20261120T0000Z",
  "tasks": [{"id": "0123456789abcdef0123", "hotkey": "owner-hotkey"}],
  "head_version": "frozen-head-version",
  "seed_secret": "private-seed-revealed-in-results",
  "budget_deadline": 1795131300.0,
  "pool_pin": "sn120-math-prior-v1:45b793b4d00d1f7225d34fd35e58018612a0ded00d96a9c23b750cfb18994115"
}
```

`tasks` is the validator's fair round-robin list of unique, verbatim 20-hex
prior-v1 IDs. There is one owner hotkey per task. Do not include the private
seed in command arguments, logs or public files.

The cache `/root/gen/pool` must contain the eight exact release shards and
`id-index.jsonl` for the pinned 811,609-row prior. `PriorPool` SHA-verifies
the manifest and shards before parsing; it never downloads during scoring.
Prepare/index the public release cache as a separate pod setup step, not as a
scorer side effect.

## Run and resume

```bash
PYTHONPATH=/root/gen/src /root/venv/bin/python -m sn120_v2.generator.scorer \
  run /root/gen/jobs/20261120T0000Z/job.json
```

The default pool cache is `/root/gen/pool`; output is
`/root/gen/jobs/<epoch>/out/`. The scorer takes a per-epoch lock. Rerunning
the same command resumes existing journals; a completed `DONE` returns its
results without inference. A different seed/head/owner/job identity is refused
rather than mixing evidence.

Sampling is production `chat_body`: temperature 1.0, top-p 0.95, top-k 20,
thinking on, max output tokens 65536, special tokens kept, one request and one
choice per rollout. The request seed is exactly
`int(sha256(secret:task_id:j).hexdigest()[:12], 16)`. The default concurrency
is 32 clients per server, assigned to the least-loaded server.

Stage 1 gives each task eight rollouts in fair owner rounds. Early decisions
come from the protocol's exact uniform-prior `q`. Undecided tasks receive
eight more rollouts, closest `q` to 0.5 first, up to 32. At 32, or when the
absolute budget deadline is reached, final acceptance uses `q >= 0.5`.
Unreached tasks and those with fewer than eight completed rollouts are
`unscored`, never accepted, and have `q: null`.

## Durable output

- `out/results.json`: epoch, head, pool, revealed seed and commitment,
  sampling parameters, per-task `s,n,q,decision`, per-hotkey counts, timings
  and throughput.
- `out/rollouts/<task_id>.jsonl`: ordered full prompt/response/grade for each
  completed rollout. Unreached tasks have an empty file.
- `out/rollouts/responses/<task_id>.jsonl`: private raw response journal for
  crash recovery. It permits out-of-order completions; only the ordered,
  graded files above determine `s` and `n`.
- `out/progress.json`: non-secret operational progress.
- `out/job-identity.json`: non-secret binding, including the seed commitment.
- `out/DONE`: written last, only after complete results and empty files for
  unreached tasks exist.

Every response and graded-prefix checkpoint is `fsync`ed. An unterminated
journal tail is discarded on resume; complete prior lines are preserved.
Grades are committed in sample order, with no duplicate inference for a
response already durably recorded. An expired job can finalize unscored
results without contacting unavailable servers.

The validator, not this process, publishes accepted-only rollouts and hashes
for rejected submissions.

## CI and selftest

```bash
PYTHONPATH=v2 python -m pytest v2/tests/test_generator_scorer.py \
  v2/tests/test_generator_scorer_recovery.py -q
PYTHONPATH=/root/gen/src /root/venv/bin/python -m sn120_v2.generator.scorer selftest
```

`selftest` starts eight fake loopback servers (automatically allocated ports)
and uses the real pinned `GradePool`. It verifies exact sampling parameters,
full and cutoff jobs, durable rollout files and zero-request resume. Use
`--root <new-empty-directory>` to retain the test outputs. `--first-port N`
is available for an explicit isolated test range; it does not alter the
production `8100..8107` contract.
