# 64k training-cap benchmark

Offline measurement harness for the **unmodified production trainer**,
`sn120_v2.gpu.train_job`. It changes only the training recipe caps; it does not
activate a parameter set, deploy code, or access production storage or the
chain.

## What it measures

`bench.py` builds the ordinary production training rows, calls `train_job.train`,
and prints a `RESULT` JSON line with wall time, per-GPU peak allocated and
reserved torch memory, trainer counters, and the resulting LoRA's SHA-256. It
also appends that record to `/root/t64/results.jsonl`. The memory figures are
per GPU, not summed across the box.

`run_matrix.sh` schedules independent single-GPU jobs across an 8xB200 box:

- `long32`, `long48`, `long64`: the same two-row group at output/context caps of
  32768/36864, 49152/53248, and 65536/69632, respectively.
- `long32_worst`, `long64_worst`: the long group at 32k and 64k with the most
  expensive miner settings: rank 64, alpha 128, and all 10 target modules.
- `b128_32`, `b128_64`: a mixed batch of 64 prompts and 128 rollouts at 32k and
  64k. `prep_data.py` reports how many rows cross 32k and the number of output
  tokens beyond that cap.
- `short32`, `short32b`, `short64`: a short-row batch under both cap sets. The
  first two repeat the same 32k run; their LoRA hashes test repeatability. For a
  batch whose rows are all below 32k, `short32` and `short64` should produce
  bit-identical LoRAs because the changed caps never bind.
- `long64_r2`: a repeat of the long 64k job.

`prep_data.py` prepares `long.json` and `batch128.json` from one local rollout
capture. `long.json` pairs a passing rollout of at least 60,000 output tokens
with a capped failing rollout. The `short.json` fixture is a separate batch in
the same format whose rows all fit under the 32k caps.

## Staging and running

The scripts are staged copies of the measurement harness and intentionally use
its absolute `/root/t64` paths. Run them only in an isolated benchmark
workspace containing the repository at `v2/`, the base model at `model/`, a
Python environment at `venv/`, the three data files, and `data/`, `out/`, and
`triton/` directories. Place these scripts at the workspace root, make
`run_matrix.sh` executable, and run:

```sh
./run_matrix.sh
```

`SN120_SRC` selects the source tree (default `/root/t64/v2`); `SN120_PARAMS`
selects the base params (default `/root/t64/v2/tools/dedup-deploy/new-params.json`).
`bench.py --gpus N --base PATH --data PATH --tag NAME --out N --ctx N` also
supports individual runs. `run_matrix.sh` uses the original supplied commands
unchanged; no benchmark is run by the params unit tests.

## Opt-in params and admission cost

`v2/ops/train64k_params.json` is the production `new-params.json` with exactly:

```text
recipe.train_max_output_tokens  32768 -> 65536
recipe.train_max_context_tokens 36864 -> 69632
max_train_token_steps           16000000 -> 22130000
```

`v2/ops/train64k_params_t3600.json` (**RECOMMENDED**) has those same three
changes plus:

```text
train_time_budget_s 2400 -> 3600
```

Neither file is primary, and `new-params.json`, the field-gate params, and the
`Recipe`/`Params` defaults are unchanged. There is no trainer-code change and
no change to `code_digest`: the trainer already uses per-layer gradient
checkpointing, chunked output logits, and one row at a time. Miner settings
cannot override the validator-owned caps.

The token-step budget is derived from the 1,380 trainable mixed-outcome-group
king rollouts in round `r000104`:

```text
1.38298   = sum(min(n + 1, 65536)) / sum(min(n + 1, 32768))
22,130,000 = 16,000,000 x 1.38298, rounded up to the nearest 10,000
```

A file with that evaluation length mix which cost 16M token-steps at 32k
therefore costs at most 22.13M at 64k. The sealed results `r000095` (six
files) and `r000040` (six files) had between 0 and 7 truncated rows per file,
so an observed file's cost rises by at most `7 x 32,768 = 229,376`. A strict
bound for every conceivable valid file, with every row a cap-hit, would
require 2x = 32M instead; 22.13M is the measured-mix budget, not that
universal guarantee. Outputs beyond 32k now cost more token-steps because
they are no longer truncated there.

The 60-minute variant is recommended because `worker.py` treats admission's
cost cap as meaning a valid batch fits `train_time_budget_s`; a batch that
received the whole time and still did not finish becomes `NO_DECISION`,
counted for its seat. `r000040` position 0 used 15.5M token-steps, 64
minibatches and one B300 GPU, and took 2293 seconds, so 2400 seconds was
already tight at 16M. At 22.13M, one GPU per LoRA and six batches, the estimate
is about 3400 seconds from `6.77k token-steps/s x 1.035` long-row cost. The
setting 3600 remains below `round_time_limit_s - promote_reserve_s =
10800 - 600 = 10200`.

## Cut-over warm-up

`warmup.py` pre-tunes FLA/Triton kernels for row lengths above 32k before the
first live 64k round. This avoids roughly 630 seconds of first-use autotuning
per GPU. FLA's l2norm kernel key is `(D, NB)`, with
`NB = cdiv(T * 16, 65536)` for q/k heads: a new class every 4096 tokens.

The tool runs one real trainer child in parallel per selected GPU with exactly
`trainer.job_env([i], work_dir)`, hence
`TRITON_CACHE_DIR=<work_dir>/triton/gpu<i>`. Warming every per-GPU directory
covers both one-GPU and two-GPU layouts, because a two-GPU job uses its first
GPU's cache directory. Synthetic jobs use the live recipe from `--params`, one
group per inclusive target length, two rollouts per group with rewards 1 and 0
(nonzero advantages), and plain repeated text tokenized to exactly
`L - prompt_tokens` output tokens.

```sh
PYTHONPATH=v2 python v2/tools/train64k/warmup.py \
  --base-dir /path/to/resident-head \
  --work-dir /path/to/box-work-dir \
  --params v2/ops/train64k_params_t3600.json \
  --min 32768 --max 69632 --stride 2048
```

`--gpus 0,1,2,3,4,5,6,7` selects GPUs explicitly; the default is every visible
GPU. The default params file is `v2/ops/train64k_params.json`. Before starting
anything, the tool queries `nvidia-smi` compute processes and refuses every
selected GPU with an active job, with a clear conflict report. It prints
per-GPU wall time, peak memory and covered length classes. Temporary jobs stay
below `work_dir/warmup/`, every temporary job directory is deleted afterward,
and adapters are discarded and never uploaded. The pure unit tests do not run
GPU work.

The documentation-only cut-over sequence is:

1. Stop the loop.
2. Set the journaled params to **recommended** `train64k_params_t3600.json`,
   or use `train64k_params.json` to retain 40 minutes.
3. Sync the validator.
4. Dry-run the weights writer.
5. Run `warmup.py` on the box, approximately ten minutes with all GPUs in
   parallel.
6. Start one loop with guarded rollback to `ab4e228c` on the same code.

These instructions are not executed by this change. No service, production
storage, R2, chain state or weights writer is invoked by the tool or tests.

## Results (8xB200 on-demand, pod noble-orbit-8b, 2026-11-22; prod trainer @ 0edaebe6 unmodified; torch 2.14.0+cu130, transformers 5.20.2, fla 0.6.0, no causal_conv1d; base = genesis MiMo-V2.6-Distill-Qwen-9B, same architecture as the head)

Peak torch memory per GPU (one row pair: a 64k+ PASS and a capped 65,536-token FAIL, truncated to the cap):

| cap (output/context) | max allocated GiB | max reserved GiB |
|---|---|---|
| 32768 / 36864 | 38.0 | 41.7 |
| 49152 / 53248 | 47.1 | 52.9 |
| 65536 / 69632 | 56.1 | 62.3 |
| 65536, rank 64 + all 10 target modules (worst miner settings) | 58.2 | 63.5 |
| 32768, rank 64 + all 10 targets | 39.5 | 43.5 |

Real 128-row batch (64 mixed-outcome r000104 tasks x 2 king rollouts; 50 rows > 32k; 2.61M trained tokens at 32k, 3.59M at 64k): peak 38/44 GiB at 32k, 56/68-72 GiB at 64k. Usable torch memory is ~171 GiB on B200 and ~267 GiB on B300, so 64k uses about a quarter of a B300. Memory grows linearly (~0.56 GiB per 1k tokens): the per-layer Replay checkpoint, chunked output layer and one-row micro-batches already make it O(T); sdpa attention never builds a T x T matrix.

Step time on that batch, warm Triton/FLA caches:

| layout | 32k step s | 64k step s | ratio |
|---|---|---|---|
| 1 GPU per LoRA (6 batches) | 307 | 436 | 1.42x |
| 2 GPUs per LoRA (<=4 batches) | 158 | 238 | 1.51x |

Trained tokens rose 1.38x, so the cost per token is +3.5% (1 GPU) / +9% (2 GPUs, worse row balance). Cold first run on a GPU: +~630 s, once, because FLA's l2norm kernels autotune per length class (key NB = cdiv(tokens*16, 65536), a new class every 4096 tokens); the result is cached in the per-GPU TRITON_CACHE_DIR.

Equivalence: a batch whose rows are all under 32k (short.json, 64 rows) gives a bit-identical adapter under the 32k and 64k recipes (sha256 0ba75530...). Every config run twice (cold and warm) gave the same adapter sha256. The rows above 32k are the only intended change.
