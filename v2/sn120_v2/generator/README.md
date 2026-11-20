# Generator protocol primitives

SN120 netuid **120**, mechanism **1**. Public name **Generator**, public page
`https://affine.io/generator`. Schema version `affine-generator-v1`.

This package is deliberately not a controller, miner, deployment unit or chain
writer. It provides independent, tested primitives for the later launch
components. It does not change any existing evaluation or payout behavior.

## Submission

A strict UTF-8 JSON object, at most **16 KiB**, with exactly these fields:

```json
{
  "version": "affine-generator-v1",
  "netuid": 120,
  "mech_id": 1,
  "epoch": "20261120T0000Z",
  "hotkey": "<SS58 Ed25519 hotkey>",
  "pool": "sn120-math-prior-v1:45b793b4d00d1f7225d34fd35e58018612a0ded00d96a9c23b750cfb18994115",
  "task_ids": ["<published 20-hex id>"],
  "signature": "<128 lowercase hex characters>"
}
```

`task_ids` contains 1..32 distinct IDs from the published pool, in their
submitted order. The ID is the release's `id` column **verbatim**. This
mechanism does not mint `mp:` IDs or ingest a second copy into the production
reserve. JSON types are strict: for example, `true` is not integer `1`.
Unknown fields, duplicate IDs, uppercase hex and empty strings are invalid.

The Ed25519 signature is over these exact UTF-8 bytes, with no trailing
newline:

```
affine-generator-v1|120|1|<epoch>|<hotkey>|<pool>|sha256(newline.join(task_ids))
```

`sha256(...)` means the lowercase hex SHA-256 of the joined IDs (no trailing
newline). The full `pool` pin above is used, not just its digest. The canonical
submission hash is SHA-256 of the complete submission's canonical JSON,
including its signature. Verification uses `sn120_v2.links.public_key` and
`valid_activation`. Registration and activation must be checked at the upload
boundary; this package never reads or writes chain state.

## Pool adapter

The default and only protocol pool is the existing GitHub release
[`sn120-math-prior-v1`](https://github.com/AffineFoundation/affine/releases/tag/sn120-math-prior-v1),
name `sn120-math-prior`, version `v1`, **811,609 rows**, eight zstd parquet
shards. Its explicit manifest URL is:

```
https://github.com/AffineFoundation/affine/releases/download/sn120-math-prior-v1/MANIFEST.json
```

The raw manifest SHA-256 is
`45b793b4d00d1f7225d34fd35e58018612a0ded00d96a9c23b750cfb18994115`.
`AFFINE_GENERATOR_POOL_MANIFEST_URL` can explicitly select another location
for the **same pinned manifest bytes** (for example, a local test/mirror URL);
it cannot silently switch pools. `AFFINE_GENERATOR_POOL_CACHE` selects the
operator's shard/index cache. There is no invented machine-specific fallback.

`pool.py` verifies the raw manifest and every shard hash before parsing.
`pyarrow` is optional (`v2[generator]`) and needed only for row reads/index
builds. The cache is ID -> shard/row, not a reprocessed dataset. `PriorPool`
returns original rows and uses `sn120_v2.envs.math_pool.Task.messages()` and
`sn120_v2.envs.math_grader.grade()` directly. Prompt and grading rules are not
reimplemented here. `public/generator/pool` is only this small manifest
pointer, never the whole pool.

## Clock and objects

An epoch is four hours, aligned to UTC midnight. Its `YYYYMMDDTHHMMZ` ID is
its **deadline / scoring start**, not its opening. `20261120T0000Z` accepts
uploads during `[2026-11-19T20:00Z, 2026-11-20T00:00Z)`. A timestamp exactly
at the deadline belongs to the next epoch. The king freezes at the deadline in
**BF16 (weights and KV cache)**; FP8 is not a permitted substitute.

All object names live in `epoch.py`:

- `public/generator/pool` — manifest pointer only.
- `public/generator/links/<hotkey>.json`.
- `private/generator/sub/<epoch>/<hotkey>.json`.
- `public/generator/epochs/<epoch>/{manifest.json,results.json,accepted.jsonl,rollouts/...}`.
- `public/generator/payout.json`.
- `state.json`.

## Sequential admission

Sampling is temperature **1.0**, top-p **0.95**, top-k **20**, maximum
**65,536 output tokens**, with thinking enabled. The prior is Beta(1,1).
After `s` passes out of `n`, `p ~ Beta(s+1,n-s+1)` and
`q = P(0.25 <= p <= 0.75)`. The standard-library implementation uses the
exact binomial-tail identity; SciPy is not required.

- At 8 samples: passes **3..5 accept**; **0 or 8 reject**; otherwise sample 8 more.
- At 16: **6..10 accept**; **<=2 or >=14 reject**; otherwise continue.
- At 24: **8..16 accept**; **<=3 or >=21 reject**; otherwise continue.
- At 32: the final **q>=0.5** rule accepts **8..24**; all others reject.
- If the budget is exhausted early, `decision(..., final=True)` applies the
  same explicit final rule: **q>=0.5 accept, q<0.5 reject**. Without this
  explicit flag, a partial count never silently becomes final.

The boundaries are inclusive. `decisions.py` exposes `q`, exact rational `q_exact`,
and structured `Decision` results; no sampling or grading is hidden in it.

## Payout primitives

Only accepted tasks enter the permanent once-ever ledger. Among valid uploads
for an unowned task, earliest store-authenticated upload wins; an exact
 timestamp tie is broken by lower file SHA-256. Every distinct accepted task
can be paid at most once, even after it leaves the trailing **24-hour** window.
Recent tasks receive equal weight per task (multiple tasks for one UID add).
Only a hotkey still registered on its earning UID is payable.

`payout_weights()` sends **all weight to UID 0** if there is no payable accepted
task, or if `updated_at` is more than **6 hours** old (a future timestamp is
also invalid). It never returns an empty or all-zero table. These are pure
primitives only: no chain weight submission or deployment is included.
