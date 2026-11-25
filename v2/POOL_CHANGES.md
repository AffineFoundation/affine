# Math pool v2.1

The public pool is now `sn120_v2/envs/math_pool_v2_1.jsonl`: 20,120 tasks (7,465 `affine_math` and 12,655 `dapo`). Its sha256 is `9c3010317c403da0f013e15d9420cab2c241d388f3325f3de8f0ad2cb5282a0a`.

The earlier pool, 21,574 tasks with sha256 `8d34d276bd777c2b48506687ee902041e40f68073ce71cf60f79b27471d06ebc`, is kept byte-for-byte at `sn120_v2/envs/archive/math_pool_v2.jsonl`.

## What changed

1,454 tasks were dropped: 1,427 `dapo` and 27 `math`. No answer key was changed. Kept JSONL lines are byte-identical and in the same order. `math:0` to `math:3`, used by admission's known-good probe, are still present.

The change follows the public third-party audit at https://github.com/catoneone/sn120-k1-fixes, pinned to commit `427295adc6b8f3d6a8c6cf66f3f00cb2d15b3122`. The two source lists and their SHA-256 values are vendored in `sn120_v2/envs/audit/`. The task IDs and answer keys were checked against the original pool. The build independently spot-checks the audit's `KEY_FORMAT` cases with the live grader. `sn120_v2/envs/tools/build_clean_pool.py` rebuilds the pool and the dropped list deterministically.

The exact rule is in `sn120_v2/envs/math_pool_v2_1.dropped.json`. In plain form:

- Start with an empty drop set `D`.
- Add the ID of every item in `v2new.remove` and `1108.remove`.
- For every item in either `fix_key` list, add the ID when its verdict is `BAD_KEY`.
- For every `KEY_FORMAT` item, grade **that item's own** `new_key` against the pool's unchanged answer with `grade_answer(gold, problem, '<think>x</think> \\boxed{' + new_key + '}', False)`. Add the ID if the code is not `P`.
- An ID is dropped if **any** of its items says drop. There is no precedence between audits.
- `kept_key_format` is the set of IDs of passing `KEY_FORMAT` items minus `D`. It has 156 IDs.

The dropped list gives the source verdicts of every removed task. Miners are told by the generated `llms.txt` on the next publish. A removed prompt is now a non-pool prompt: it has no official grade and its rewards are not compared. The normal `min_pool_prompt_frac` rule still applies.

## Deploy

Deploy a new checkout on the GPU box **between rounds**. Nothing in `sn120_v2/gpu/*.py` changed, so its `code_digest` does not change. Default params and `params_hash` do not change.

At startup, a non-empty `params.env_pin` must equal the pin of this checkout's pool, grader source and pinned grader packages. Print it on the new checkout, from `v2/`:

```
python -m sn120_v2.envs.math_grader --env-pin
```

Setting `env_pin` to that value is recommended. It changes `params_hash`; use the normal `set-params` procedure with the complete parameter file. An empty pin keeps the previous behavior.

The new pool changes the IRT table at the next start. Rows of kept tasks retain difficulty, information and rounds seen; removed rows disappear. Sampling masses are recomputed and the next round uses a new snapshot ID. Closed rounds remain valid journal records. The first four admission probe tasks are unchanged.

## Replicate an earlier round

From `v2/`, use the normal command in [REPLICATE.md](REPLICATE.md). `replicate run` now defaults to `--pool auto`: if any task of the recorded draw is missing from the current pool, it uses the archived v2.0 pool and its unchanged keys. The report names this under `differences`.

Use `--pool v2.0` to force the archive, or `--pool current` to refuse a draw that needs a removed task. Explicit Python callers can use `load_pool(POOL_V2_0_FILE, POOL_V2_0_SHA256)` and pass that pool to `replicate.parts`. Do not use the old `math_pool_v2.jsonl` outside `archive/`; it is retained only as a source of the v2.0 archive.

## Continuous public-reserve rotation (not enabled)

[POOL_ROTATION.md](POOL_ROTATION.md) adds an append-only audited reserve, ingest/withdraw tools, and a deterministic `k`-capped active-set rotation at round open. It is off by default: no reserve bytes are read or written, default and prod-shaped params hashes stay unchanged, and off-mode IRT, journal, round and publisher bytes stay unchanged. Activation needs a separate normal `set-params` with `rotation_enabled: true` between rounds and Jacob's approval. Future audited additions belong in the reserve rather than another v2.1-style pool-file swap.
