# Affine benchmarks (`bench/`)

Public math benchmarks for every SN120 king and for reference ~8-9B reasoning models, all re-run on one rented GPU node with
the same vLLM endpoint, prompts, graders and sampling. Results: <https://affine.io/benchmarks.json>; table at the top of
<https://affine.io>. Informational only: nothing in the gate, promotions, weights or the trainer reads it.

```
host (affinevalidator)                              node (Lium GPU pod, e.g. affine-bench-1 8xB200)
  affine-bench daemon  (systemd user unit)            /data/bench/repo        this repo (git clone)
   - king list: affine.io/performance.json            /data/bench/venv-vllm   vLLM 0.32.0 (deploy/requirements-vllm.txt)
   - queue: newest king > new GREEN kings (<6 h)      /data/bench/venv-vf     verifiers 0.4.0 + prime 0.19.0 + envs
     > Genesis + all refs > GREEN backlog > rest      deploy/serve.sh         ONE OpenAI endpoint :8000/v1 (model "bench"),
   - weights: king = R2 presigned GET piped in          data-parallel = one replica per GPU
     (deploy/gate_fetch.py, sha256-checked);          affine-bench node-run   vf-eval per benchmark, all at once
     refs = public HF at a pinned commit                -> runs/<model>/<bench>/{vf/<run>/traces.jsonl, summary.json}
   - collects summary.json -> state/bench-results/
   - publishes affine.io/benchmarks.json (aggregates only; no per-task MATH-test data, ever)
```

* Benchmarks: `config/benchmarks.toml` (registry: id, suite, verifiers env, pinned source, rollouts, enabled). Adding a
  benchmark (SWE-bench Verified, Terminal-Bench 2, BrowseComp, tau2-bench are on the Prime Hub) is a block there plus a
  pinned line in `deploy/requirements-envs.lock` (and a `harness`/`runtime` that suits it).
* Models: `config/models.toml` (Genesis + references, pinned HF commits). Kings come from performance.json.
* Serving + sampling: `config/serving.toml` (temperature 0.6, top_p 0.95, top_k off, 32,768 max new tokens, thinking on).
  Results carry `bench_version` = `<bench>/v<version>/<serving version>`; bump a version to rescore.
* Envs: Prime Hub `primeintellect/{math500,aime24,aime25,aime26,gsm8k}` pinned by wheel sha256, wrapped by
  `environments/local-*` only to open the sandbox network policy for the local subprocess runtime (prompt, data, order and
  reward unchanged); own envs `aime22`, `aime23`, `amc23`, `math-test` (full MATH test, 5,000) in the same format and
  reward (`vf.verify_boxed_math_answer`); `olympiadbench` + `omni-math` use `affine-math-grader` (tests in `tests/`) and are
  disabled for now. Harness = verifiers `null` (one chat turn, no tools).

## Use

```
# host: one model x benchmarks through the daemon's queue (runs before everything else)
PYTHONPATH=~/bench-work/affine/bench python -m affine_bench.cli run --model king:7ca757b2 --benchmarks math500
PYTHONPATH=~/bench-work/affine/bench python -m affine_bench.cli run --model qwen3-8b --benchmarks aime24 --rollouts aime24=32 --tag try1
PYTHONPATH=~/bench-work/affine/bench python -m affine_bench.cli status
# node: directly against the endpoint, any harness
. /data/bench/env.sh; vf-eval local-math500 -m bench --client.base-url http://127.0.0.1:8000/v1 --client.api-key-var BENCH_API_KEY \
   --env.agent.harness.id null --env.agent.runtime.type subprocess -r 1 --no-push --sampling.temperature 0.6 --sampling.top-p 0.95 --sampling.max-tokens 32768
```

Rebuild from scratch: [REDEPLOY.md](REDEPLOY.md).

## Disk policy

Weights are never deleted automatically. The pod has about 21 TB, and every king plus reference needs
about 3.5 TB in total. `daemon.prune` deletes only when `BENCH_PRUNE=1` is set, and that needs the owner's approval.
