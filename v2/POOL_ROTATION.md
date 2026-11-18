# Continuous task rotation (off by default)

**Not merged, not deployed. Enabling needs Jacob's go.** Merging this code changes no round, pool, params hash, environment pin or GPU `code_digest`. Production activation is a separate `set-params` between rounds.

## Reserve layout

`sn120_v2/envs/reserve/` holds:

- `shards/<NNNN>-<slug>.jsonl`: immutable, canonical JSONL. Each row has `id`, `env`, `problem`, `answer`, `source` (dataset, revision and original row), `license`, `added_at` (the UTC `YYYY-MM-DD` ingest date) and `audit`. Only `audit.verdict == "keep"` is accepted. `math:` and `dapo:` prefixes are reserved for the base pool.
- `withdrawn.jsonl`: append-only `{id, reason, added_at}`. A withdrawn reserve task is not admitted; a withdrawn active task is retired outside the `k` cap.
- `MANIFEST.jsonl`: shard rows `{shard, sha256, count, added_at}` in admission order, followed by the current `{withdrawn: {sha256, count}}` pointer. `reserve_manifest_sha` is the SHA-256 of this file's bytes.

The shipped reserve has zero shards. The loader checks every listed shard's bytes, canonical schema, unique ids and normalised problems. A later round must extend the shard history in the same positions and may only append withdrawals. Tampering fails closed with `ConfigError`; no round opens. `SN120_RESERVE_DIR` is a test-only override.

The reserve is not in `env_pin` or `params_hash`. Each rotated round records the reserve manifest SHA in its journal, round file, public rotation artifact and IRT snapshot. Shards and the manifest are mirrored under `public/pool/reserve/`; miners need no git checkout. The active set is public before its round: `public/pool/active/<active_sha>.json`; the next-up queue is in each round's `pool` block. On a fresh checkout, `PoolRuntime` restores the mirrored shards and retired archive before reading the catalog; a newer local manifest may extend, but not replace, that history.

## Ingest

From `v2/`, with the same grader extras used by `build_clean_pool`:

```
python -m sn120_v2.envs.reserve ingest \
  --source omni-math@<revision> --license CC-BY-4.0 --prefix omni \
  --audit-method <method> --audit-ref <audit-report-or-commit> input.jsonl
python -m sn120_v2.envs.reserve verify
python -m sn120_v2.envs.reserve status
python -m sn120_v2.envs.reserve withdraw omni:17 --reason 'wrong answer key'
```

Input has one `{problem, answer, source_id?}` per line. `--env` chooses the task slice (default `affine_math`). Ingest NFKC-normalises, case-folds and whitespace-collapses problems for dedupe. It checks against the v2.1 pool, v2.0 archive, all reserve shards, withdrawn ids, the retired archive and the input itself. It uses the live grader self-check `grade_answer(answer, problem, '<think>x</think> \\boxed{' + answer + '}', False) == P`, requires a nonempty single-line answer, rejects MANDATE, and enforces params-free problem/answer bounds of 16,384/2,048 characters. `--reject-mc` also applies the grader's multiple-choice heuristic. Rejected rows and counts go to `<shard>.rejected.json`, never to the reserve.

IDs continue at `<prefix>:<max existing n + 1>`. An empty re-run writes nothing. Re-running after new rows have been ingested creates no duplicate and leaves already listed shards byte-identical. Repeating the same source/input filename with different content fails closed rather than allocating a colliding source shard. Only the UTC ingest date is time-dependent; with the same date, input and reserve the output is byte-identical. Ingest never changes the active set.

A high-retired task returns only explicitly:

```
python -m sn120_v2.envs.reserve ingest ... --readmit omni:17 readmit.jsonl
```

Its `source_id` must be the retired id, problem and answer must exactly match the archive, and the row carries `audit.readmit_of = <retire round>`. It is a new queue entry. Low-retired and withdrawn tasks cannot return.

## Rule (`affine-pool-rotation-v1`)

At round open, before the draw, use the previous close's IRT snapshot and previous rotated active set `A_prev`. The first enabled round uses the 20,120 base-pool ids and an empty ledger. `A_initial` is the base-pool id set.

- `W`: active ids in the withdrawn list. All leave with reason `withdrawn`, outside `k`.
- `H`: ids in `A_prev \\ W` with `rounds_seen >= rotation_n_min` and `sigmoid(head_ability - difficulty) > rotation_retire_p`. `head_ability` is the snapshot's own `head_version_id` ability, exactly `public_snapshot`'s value (zero when absent).
- `L`: when `rotation_retire_low_p > 0`, ids with at least `rotation_n_min_low` null-M rollouts and empirical `passes / rollouts < rotation_retire_low_p`. Resolved `P`, `F`, `C` count; `C` is a fail. Infrastructure and unrun outcomes do not count. High eligibility takes precedence for a task in both sets.
- `Q`: reserve rows in shard order; within a shard, ascending `sha256(<shard_sha256>|reserve-order|<task_id>)`, minus consumed original admissions and withdrawn ids. Explicit readmits are new entries.

Order `H` by descending IRT p, then `L` by ascending empirical rate, with `sha256(<round_id>|retire|<task_id>)` breaking ties. With `target = rotation_active_size or |A_initial|` and `avail = |Q|`:

```
n_retire = min(k, |H union L|, avail + max(0, |A_prev \\ W| - target))
n_admit = clamp(target - (|A_prev| - |W| - n_retire),
                0, min(avail, k + |W|))
A_r = sorted((A_prev \\ W \\ retired) union Q[:n_admit])
```

No high/low retirement happens with an empty queue. An above-target active set may shrink toward target without reserve admissions. Withdrawals can cause an otherwise empty reserve to change the set. `next_up` is the following `2*k` queue ids.

The complete decision, its inputs, the ledger, last IRT rows and archive are public at `public/pool/rotation/<round_id>.json`. Its canonical hash is `state_sha`; `public/pool/state/<state_sha>.json` has the same bytes. `public/pool/retired.json` is the cumulative append-only-in-content archive; `retired_tasks.json` keeps complete Task rows for grading and exact readmission.

## IRT and ledger

An admitted id starts at `(round(head_ability, 6), PRIOR_PRECISION, 0)`, so p=0.5 and Fisher `p(1-p)` is maximal. The uniform 10% share is unchanged. Other new pool ids keep the old `(0.0, PRIOR_PRECISION, 0)` prior, including PR-276-style base-pool changes.

The snapshot remains `affine-irt-v1`. A `rotation = {active_sha, reserve_manifest_sha, round_id}` **body field** is present only when rotation is enabled and is included in `snapshot_id`. Old and off-mode snapshots have no such field and keep their exact ids and verification/draw behavior.

At close, null-M resolved outcomes update `(head_rollouts, head_passes)` since admission. This state is in the content-addressed rotation artifact, not a giant repeated journal record. The journal carries the active SHA and artifact hashes. Artifacts precede ROUND_OPENED. Restarting an open round reads its exact stored table and active file and performs no second rotation; replay mismatch is `VerifyError`. A crash before the journal record is idempotent because identical inputs write identical artifacts.

## Recompute and replicate

```
python -m sn120_v2.replicate rotation \
  --mining-json mining.json --round r000123 --work /data/affine
```

The command downloads the public decision, previous state and source IRT snapshot and compares the complete recomputed Rotation. `replicate run` uses the rotated round's active artifact and mirrored reserve for its catalog. No file under `sn120_v2/gpu/` changes.

## Enable (only with Jacob's approval)

1. Ingest and `verify` a reserve shard. Commit only the shard and manifest line; the live box reads it on the next round open after the normal code update.
2. Stop `run` between rounds and use normal `set-params` with a complete params file containing `rotation_enabled: true`. `rotation_k=32`, `rotation_retire_p=0.95`, `rotation_n_min=3`, low-retire off and `rotation_active_size=0` are the defaults.
3. Restart `run`. The first complete new round rotates before its draw. Existing open rounds are never changed in flight. Keep monitoring active size, retire/admit ids, queue and ledger; rollback is a normal between-round params change with `rotation_enabled: false`.
