# SN120 trusted paired sampler

An independent check of whether each training step on the SN120 MiMo mainnet run made the model better.
It is fully detached from miners, the trainer, the validator and the weight writer. It signs nothing, sends
no chain transactions, uses no credentials, and makes no calls to the network's verifiers.

For each published consecutive checkpoint pair (before = input of training epoch e, after = its output) one
**sampler instance** does the following:

1. Draws **its own** random K tasks, uniformly and without replacement, from the full RL math pool. The pool is the
   6,746 `affine_math` indices offered to miners (`OPENING.payload.environments[affine_math].indices`, task snapshot
   `assets/original-math7496.tasks.json`). The draw uses a fresh CSPRNG seed (`secrets.randbits`), so there is no
   coordination between instances. The seed is recorded.
2. Samples m completions per task from **both** checkpoints. Settings match the network's harness: chat template with
   `enable_thinking`, T=1.0, top_p=0.95, top_k=20, max_tokens=min(8192, 12288−prompt), EOS {248046, 248044}, bf16.
   Generation uses vLLM with a per-request `SamplingParams(seed=…)`. **Common random numbers (default):** sample j of
   task t uses `seed = int(sha256(f"{rng_seed}:{t}:{j}")[:8], 16) & 0x7fffffff` for **both** checkpoints, so the two
   sides see the same random stream. `--no-crn` appends `":after"` to the after side's hash input, which gives
   independent streams. The rule is written to every record (`generation.common_random_numbers`,
   `generation.seed_scheme`, `tasks[].seeds_before/seeds_after`). The score cache is keyed by
   `(checkpoint, task, seeds, max_tokens)`, so a cached score is only reused for exactly the same random stream. Seeds
   derive from the instance's own `rng_seed`, so there is no cross-instance coordination. Records written before the
   CRN change (backfill and `g0`) used a per-instance `getrandbits(31)` list, shared by both sides
   (`common_random_seeds: true`, `tasks[].seeds`); that is equivalent to CRN.
3. Grades every sample locally with a pure-Python replica of the network's outcome rule (`grader.py`):
   - mimo-final-channel projection: exactly one `<think>…</think>`, no control tokens, and the final channel holds
     exactly one balanced `\boxed{}`. A budget stop counts as unresolved, i.e. 0.
   - math-verify 0.9.0 `parse`/`verify` against `\boxed{gold}`, as in `affine_math_v1/verify.py`.
   - Grader errors are `indeterminate`: they are excluded, never scored 0.

   Raw correctness is recorded separately from the penalized reward. The penalized reward applies the live v8.2.4
   overlong soft penalty: `correct − 0.5·clip((len−6000)/2192, 0, 1)`.
4. Appends one JSONL record per (instance, pair) to `results/<sampler_id>.jsonl`. The record holds ids/steps, the rng
   seed, the task ids and seeds, per-task before/after mean correctness, reward and length, the outcome counts, the
   generation settings, the paired delta ± SE (plus unpaired SE for comparison) and per-phase timings.

The only shared state between instances is:

- the read-only checkpoint index;
- the append-only `results/` directory, one file per instance;
- a content-addressed checkpoint cache. Every file is sha256-checked against the signed filemap, and the filemap must
  hash to the checkpoint id before use.

## Pieces
| file | where | what |
|---|---|---|
| `export_checkpoints.py` | Arbos `~/subnet120-rewrite/tools/trusted-sampler/` | Read-only. Scans `state/mimo-mainnet-v1/epochs/*/coordinator/OPENING.json` and writes `index.json`: checkpoint id, filemap, input checkpoint, step, the presigned **GET** URLs already published to miners (~24 h expiry), and the pool. Copies no miner data. `--watch 120 --push host:path` re-exports every 2 min and scp's the index to sampler hosts. |
| `fetch.py` | sampler host | Downloads and authenticates checkpoints into `/root/ckpt-cache/<id>/`. Lock-shared by instances. |
| `grader.py` | sampler host | The local outcome-rule replica. |
| `sampler.py` | sampler host | The instance. Modes: `--pair A:B`, `--backfill N`, `--watch`. Optimizations: `--engine swap` (resident vLLM engine, weights hot-swapped in place) and `--chain` (each checkpoint generated once for both adjacent pairs). |
| `aggregate.py` | anywhere | Pools all records per pair (task-level) and prints pooled delta ± SE (paired and unpaired). |
| `launch.sh` | sampler host | One instance per GPU. |
| `extract_inclusion.py`, `backtest.py` | Arbos / sampler | Backtest: per-epoch UID inclusion → per-miner credit regression (`REPORT.md`). |
| `crn_compare.py` | anywhere | Paired SE with vs without CRN per pair at K=128. |
| `combined.py`, `run_combined.sh`, `combined_analyze.py` | sampler host | Four-way experiment (independent / CRN / log-prob-IS / combined) on the same tasks; `run_combined.sh GPU PAIR ID` waits for an empty GPU first. |
| `logprob_proto.py` | sampler host | Prototype of log-prob scoring: before-samples re-scored under the after checkpoint (prefill only). Writes `results-logprob/`. |
| `test_swap.py`, `test_swap2.py`, `bench.py` | sampler host | Swap correctness (param-level: 364/364 tensors bitwise equal to a cold load) and throughput sweep. |

## Launch
```bash
# on the sampler pod (root@<SCORER_HOST> -p <SCORER_SSH_PORT>), code in /root/trusted-sampler, venv /root/sampler-venv
./launch.sh 8 --watch --chain --engine swap -K 256 -m 8   # recommended live config (one instance per GPU)
./launch.sh 8                                   # plain default: --watch -K 256 -m 8 (reload engine, no chain)
./launch.sh 4 --backfill 6 --chain -K 256 -m 8  # backfill the 6 newest pairs on GPUs 0-3
GPU_OFFSET=4 ./launch.sh 4 --pair cf791667:19f21658 -K 128 -m 8
/root/sampler-venv/bin/python aggregate.py results/          # pooled table (add --metric reward / --json)
```
Instances start `STAGGER` seconds apart (default 20). Concurrent first-time vLLM/Inductor compile-cache writes can
race and kill an engine at start-up (`FileNotFoundError … inductor_cache`). If one dies, just start it again on its own;
use `STAGGER=60` when starting several instances with new shapes or flags.

`--chain` (watch/backfill) generates each checkpoint once, on the after-half of pair (e−1→e) together with the
before-half of pair (e→e+1), with a fresh task draw per pair. `--engine swap` keeps one engine resident and loads
the next checkpoint's weights in place (~22 s instead of an ~85–130 s engine rebuild).

Adding an instance (on this or any other GPU host with the venv, the code and an index copy) just adds records. To add
a new host, add another `--push` target to the exporter on Arbos.

The index exporter runs on Arbos as a detached process (`nice 19`, read-only):
```bash
cd ~/subnet120-rewrite/tools/trusted-sampler && setsid nohup nice -n 19 python3 export_checkpoints.py --out index.json \
  --watch 120 --push root@<SCORER_HOST>:/root/trusted-sampler/data/index.json \
  --ssh-opts "-o BatchMode=yes -o UserKnownHostsFile=$HOME/.lium/known_hosts/21eefe00-39e9-4fb3-88fc-2f565eb0f5eb -P <SCORER_SSH_PORT>" > export.log 2>&1 &
```

## Caveats
- Log-prob (importance-weighted) scoring is **biased** as implemented (it uses untruncated log-probs, while
  sampling used top-k/top-p); see REPORT.md §3. Use generation scoring.
- Killing a sampler with plain `kill PID` can orphan its `VLLM::EngineCore`, which keeps holding the GPU. The next
  engine on that GPU then fails at start-up ("Free memory … less than desired"). Kill the process group instead:
  `kill -- -PGID`. Every instance is started with `setsid`, so PGID = PID.
- Backfill is limited to checkpoints whose presigned URLs are still valid (about 24 h after their epoch opened) or
  that are already in the cache. Older pairs need a read-only R2 credential, which does not exist yet.
- Sampling is distributionally identical to the network's (same T/top-p/top-k and template) but not bit-identical to
  its inverse-CDF sampler. The grader is a faithful replica. It is not the network's isolated, runtime-locked grader
  process.
- The padded LM-head tokens (id ≥ 248077) are not masked. If one is ever sampled, that sample counts as unresolved.
