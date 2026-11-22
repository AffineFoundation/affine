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

`v2/ops/train64k_params.json` is the production `new-params.json` with only:

```text
recipe.train_max_output_tokens  32768 -> 65536
recipe.train_max_context_tokens 36864 -> 69632
```

It is not the primary parameter file. There is no trainer-code change and no
change to `code_digest`: the trainer already uses per-layer gradient
checkpointing, chunked output logits, and one row at a time.

`max_train_token_steps` stays at 16,000,000, preserving the bound on counted
token-step work per file. Outputs beyond 32k now cost more token-steps because
they are no longer truncated there. Consequently, a file with rows beyond 32k
that was near the cost cap and valid under the 32k recipe can exceed the cap
and become invalid under this recipe. Miner settings still cannot override
these two validator-owned caps.

## Results

Pending final measurements from the 8xB200 benchmark box.
