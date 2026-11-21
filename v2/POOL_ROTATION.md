# Continuous task rotation (off by default)

**Not merged, not deployed. Enabling needs Jacob's go.** Merging this code changes no round, pool, params hash, environment pin or GPU `code_digest`. Production activation is a separate `set-params` between rounds.

## Reserve layout

`sn120_v2/envs/reserve/` holds:

- `shards/<NNNN>-<slug>.jsonl`: immutable, canonical JSONL. Each row has `id`, `env`, `problem`, `answer`, `source` (dataset, revision and original row), `license`, `added_at` (the UTC `YYYY-MM-DD` ingest date) and `audit`. Only `audit.verdict == "keep"` is accepted. `math:` and `dapo:` prefixes are reserved for the base pool.
- `withdrawn.jsonl`: append-only `{id, reason, added_at}`. A withdrawn reserve task is not admitted; a withdrawn active task is retired outside the `k` cap.
- `MANIFEST.jsonl`: shard rows `{shard, sha256, count, added_at}` in admission order, followed by the current `{withdrawn: {sha256, count}}` pointer. `reserve_manifest_sha` is the SHA-256 of this file's bytes.

The shipped reserve is an immutable packaged seed. The loader checks every listed shard's bytes, canonical schema, unique ids and normalised problems. A later round must extend the shard history in the same positions and may only append withdrawals. Tampering fails closed with `ConfigError`; no round opens.

When `SN120_RESERVE_DIR` names a live directory, `PoolRuntime` seeds it from the packaged directory on first startup, including any untracked retired archive. Live shard history must extend the packaged prefix. If a later packaged deploy adds a suffix to a strict-prefix live directory, its new shards are copied forward; divergence is refused. Startup and each open use the shared/exclusive `<live>/.ingest.lock` with ingestion. An append-only, fully verified new manifest reloads before rotation. A bad candidate logs one line and keeps the last verified reserve. Without the variable, packaged behavior is unchanged.

Production's run loop and `systemd/generator-ingestd.sh` read the absolute live directory from one line in `/root/sn120-run/reserve-live.path`; the run loop exports it as `SN120_RESERVE_DIR`. If that file is absent the run loop is unchanged. A typical deployment writes `/root/sn120-run/reserve-live` to this file. No environment setting is needed when the file is absent.

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

Input has one `{problem, answer, source_id?, prior_p?}` per line. `prior_p` is an optional finite public prior pass rate in [0,1], used by the ranked v2 rule when no IRT row is available. `--env` chooses the task slice (default `affine_math`). Ingest NFKC-normalises, case-folds and whitespace-collapses problems for dedupe. It checks against the v2.1 pool, v2.0 archive, all reserve shards, withdrawn ids, the retired archive and the input itself. It uses the live grader self-check `grade_answer(answer, problem, '<think>x</think> \\boxed{' + answer + '}', False) == P`, requires a nonempty single-line answer, rejects MANDATE, and enforces params-free problem/answer bounds of 16,384/2,048 characters. `--reject-mc` also applies the grader's multiple-choice heuristic. Rejected rows and counts go to `<shard>.rejected.json`, never to the reserve.

IDs continue at `<prefix>:<max existing n + 1>`. An empty re-run writes nothing. Re-running after new rows have been ingested creates no duplicate and leaves already listed shards byte-identical. Repeating the same source/input filename with different content fails closed rather than allocating a colliding source shard. Only the UTC ingest date is time-dependent; with the same date, input and reserve the output is byte-identical. Ingest never changes the active set.

A high-retired task returns only explicitly:

```
python -m sn120_v2.envs.reserve ingest ... --readmit omni:17 readmit.jsonl
```

Its `source_id` must be the retired id, problem and answer must exactly match the archive, and the row carries `audit.readmit_of = <retire round>`. It is a new queue entry. Low-retired and withdrawn tasks cannot return.

## Generator accepted-task feed

Build the compact prior index once, in a tooling environment with `pyarrow` (install the `prior_index` extra), from the checksum-verified eight-parquet `sn120-math-prior-v1` release:

```
python -m sn120_v2.envs.tools.prior_index /path/to/prior-release /root/sn120-run/generator/prior-index.json
```

The index maps `sha256(normalise(problem))` to the prior's `source` (`dataset@revision`) and `license`; it is JSON, so production ingestion needs no `pyarrow`. The manifest and all eight shards are SHA-checked while building it. Ingest an epoch with the public result document:

```
python -m sn120_v2.envs.reserve ingest-generator accepted-EPOCH.jsonl \
  --epoch EPOCH --results results-EPOCH.json \
  --prior-index /root/sn120-run/generator/prior-index.json
```

The default active-set dedupe input is fetched from `SN120_PUBLIC_URL` (default `https://affine.io`); `--active` supplies a verified public id list for offline use. Completeness is `.done` first, then `results.json`; without either, ingestion refuses with exit 3. A marker is `accepted-EPOCH.done`, either `{"rows":N,"sha256":"hex"}` (extra keys such as `epoch` allowed) or one plain `N hex` line. A wrong row count or SHA exits 3 without writing. `results.json` checks epoch, accepted count and prior-pool SHA. Public v1 result files currently name their schema marker `version`, not `schema`. Accepted rows must include the exact `s/8` prior. Row `task_id` replaces `source_id` in Generator provenance when present. Row `license` takes precedence over the prior-index join; `source`/`license` pairs take precedence over the join for older files. `prior_king` or `king_revision` is recorded in audit and ledger, falling back to the result's king. A missing index match without row provenance is `not_in_prior`; NC, ND and unrecognized licenses are rejected.

The proof filter is deliberately small: reject `prove`, `show`/`show that`, `demonstrate` or `证明` when there is no value request (`find`, `determine`, `calculate`, `compute`, `evaluate`, `value`, `solve for`, `how many/much`, `what is`, `求`, `计算`, `多少`, `值`). Normal dedupe covers active, base/archive, reserve, retired, same-file and earlier-epoch rows. The existing length, MANDATE, MC and grader self-check filters also run. IDs are continuous `gen:<n>`; shard names are `NNNN-sn120-generator-<lowercase epoch>-<license slug>.jsonl`, with no extra dots in the slug. Generator `audit.verdict` is the existing accepted value `keep`; the audit method/ref records the frozen-king band and public result URL.

`<live>/generator-ingested.jsonl` has one row per epoch, including zero-kept epochs. The same input SHA is an idempotent no-op; another SHA exits 4 unless it is a recognized backfill, and an incomplete input/result/marker exits 3, both without a reserve write. The ledger records the normalized problem set. A backfilled file whose problems are a subset of that set adds a `backfill_seen` note and does not ingest. Genuinely new problems go through the normal dedupe/filter path into `-b1` shards; the single epoch ledger row accumulates history and backfill notes. Shards are verified in a staged generation before the manifest commit. The ledger is written last, atomically, under the exclusive lock. If a crash leaves a committed manifest without its local ledger, the exact same input can recover the ledger only when every recomputed shard name and SHA matches the listed history.

Run the daemon from `v2/`, with its Python environment and the index:

```
systemd/generator-ingestd.sh /path/to/venv/bin/python \
  /root/sn120-run/generator/prior-index.json
```

It polls `/root/sn120-run/generator/inbox/accepted-EPOCH.jsonl` every 120 seconds. If `accepted-EPOCH.done` exists it takes precedence; otherwise the daemon fetches `$SN120_PUBLIC_URL/generator/epochs/EPOCH/results.json` with retries. It never ingests without one of those completeness proofs, and logs one JSON line per epoch to `/root/sn120-run/generator/ingestd.log`. Both the daemon and run loop read the same `/root/sn120-run/reserve-live.path`. Start `PoolRuntime` once before the daemon to initialize `/root/sn120-run/reserve-live`. `SN120_GENERATOR_INBOX`, `SN120_GENERATOR_LOG`, `SN120_GENERATOR_POLL_S`, `SN120_GENERATOR_PRIOR_INDEX`, `SN120_GENERATOR_LOCK` and `SN120_RESERVE_PATH_FILE` are scratch/test overrides; production defaults are the paths above. The daemon does not publish: the next ordinary `publish_open` publishes the manifest and every shard, with readback verification.

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

The complete decision, its inputs, the opening ledger, last IRT rows and archive are public at `public/pool/rotation/<round_id>.json`. Its canonical hash is the opening `state_sha`. At close, the controller stores the ledger-updated state at `public/pool/state/<state_sha>.json` in its object store; the static public site does not serve that key. Public replay reconstructs and digest-checks the previous close from the named rotation, draw snapshot, round report and king outcomes, never requiring public `pool/state`. `public/pool/retired.json` is the cumulative append-only-in-content archive; `retired_tasks.json` keeps complete Task rows for grading and exact readmission.

## IRT and ledger

An admitted id starts at `(round(head_ability, 6), PRIOR_PRECISION, 0)`, so p=0.5 and Fisher `p(1-p)` is maximal. The uniform 10% share is unchanged. Other new pool ids keep the old `(0.0, PRIOR_PRECISION, 0)` prior, including PR-276-style base-pool changes.

The snapshot remains `affine-irt-v1`. A `rotation = {active_sha, reserve_manifest_sha, round_id}` **body field** is present only when rotation is enabled and is included in `snapshot_id`. Old and off-mode snapshots have no such field and keep their exact ids and verification/draw behavior.

At close, null-M resolved outcomes update `(head_rollouts, head_passes)` since admission. This state is in the content-addressed rotation artifact, not a giant repeated journal record. The journal carries the active SHA and artifact hashes. Artifacts precede ROUND_OPENED. Restarting an open round reads its exact stored table and active file and performs no second rotation; replay mismatch is `VerifyError`. A crash before the journal record is idempotent because identical inputs write identical artifacts.

## Ranked active pool (`affine-pool-rotation-v2`)

Select this rule with `rotation_rule="info"`. The existing v1 rule and its bytes remain the default; selecting the new rule also needs `rotation_enabled: true`. The new default-omitted params are:

- `rotation_rule` (`"v1"` or `"info"`),
- `rotation_churn_frac` (default 0.10),
- `rotation_prior_p` (default 0.5),
- `rotation_queue_publish` (default 2000), and
- `rotation_churn_max` (default 0, no extra cap; when positive, an additional operational ceiling on steady-state swaps).

The candidates are the previous active set minus withdrawals, the unconsumed reserve queue, and the non-withdrawn retired archive. Withdrawn tasks never return. A v2 retirement is not permanent: a task can become a high-information candidate again. A returning task restores its exact archived `(difficulty, information, rounds_seen)` row and its head-ledger evidence; a brand-new task uses the same head-ability prior as v1. The IRT table keys remain exactly the active set.

For each candidate, score `p * (1-p)`, the Fisher information of a Bernoulli outcome. An active or archived task with `rounds_seen >= rotation_n_min` uses `predicted_pass(head_ability, difficulty)`. Otherwise use the public reserve row's `prior_p` if present, else `rotation_prior_p`. **Saturated tasks keep their score**, even when `p > rotation_retire_p`; `high` is their retirement reason. Only a **dead** task (`head_rollouts >= rotation_n_min_low` and `head_passes == 0`) is forced to score zero. `low` is its retirement reason.

With `T = rotation_active_size or the initial active size`, apply this order every round:

1. Remove withdrawn active tasks, outside the churn cap.
2. If above target, retire the lowest-scored `|A|-T` active tasks, with no cap. Reasons are `excess`, or `high`/`low` for flagged tasks.
3. If below target, admit the highest-scored outside candidates, with no cap.
4. While a strictly higher-scored outside candidate exists, swap it for the lowest-scored active task, at most `floor(rotation_churn_frac * T)` times (and additionally at most `rotation_churn_max` when that is positive). Reasons are `high`, `low`, or `displaced`. Thus the swap count is reserve-bounded: it never exceeds the number of outside candidates strictly better than the active task replaced, and it is zero when no such candidate exists. Shrink/fill and withdrawals are not swaps and are not capped by these settings.

Retirements are ascending by score; admissions are descending. Every tie uses `sha256(f"{round_id}|rank|{task_id}")`, with task ID as the final deterministic guard. There is no random number, wall-clock time, Python `hash()`, or dependence on shard/input order. `next_up` is the top `rotation_queue_publish` remaining outside candidates in rank order, not FIFO order. Reserve metadata is read once at runtime construction; the per-round selection is `O(|C| log |C|)` and carries only IDs, priors, IRT rows and ledger counts.

### Public v2 files and replay

In addition to the existing active, rotation, reserve and retired files:

- `public/pool/ranking/<round_id>.json` contains every candidate: `id`, previous `state`, `p`, `rounds_seen`, `score`, `flags`, `source_prefix`, `env`, `head_rollouts`, `head_passes`, and `admitted`/`retired`/`retire_reason` for this decision.
- `public/pool/rotation/<round_id>.json` carries the same complete candidate ranking plus admitted IDs, retired IDs/reasons, `next_up`, `rule_params` and `retire_reason_counts`.
- `public/pool/summary.json` aggregates the post-decision active composition and score histogram, reserve/retired counts, and the last 20 v2 rounds for the static `/pool` page (`pool.html`). The page has a top-50 upcoming queue and a client-side search over the complete latest ranking. All times are UTC.

The first v1-to-v2 transition publishes the complete previous retired archive, candidate metadata and retired ledger it used. `python -m sn120_v2.replicate rotation --mining-json <mining.json> --round rNNNNNN` fetches the public decision, ranking and source snapshot, reconstructs the previous close from public round evidence, and compares the complete recomputed `Rotation`, including the full ranking. The early live v1 `r000066` state is byte-SHA addressed; the later `r000069` state uses the canonical digest. Public-only replay reconstructs and authenticates both the early byte-SHA state encoding and canonical-digest rounds; new v2 states always use the canonical digest.

Strict replayer schema additions (all optional and omitted on v1/old journals):

- `Params`: `rotation_rule`, `rotation_churn_frac`, `rotation_prior_p`, `rotation_queue_publish`, `rotation_churn_max`.
- `Reserve`: `candidate_meta` (per-ID `source_prefix`, `env`, optional `prior_p`).
- `Rotation`: `candidate_ranking`, `retire_reason_counts`, `restored_irt_rows`, `initial_retired_archive`, `initial_candidate_meta`, `initial_retired_ledger`, `retired_ledger`, `opened_at`.
- Journal: no new record kind and no new `RoundRecord` field. The existing optional `RoundRecord.rotation` map additionally carries `retire_reason_counts` and the v2 `rule_version`; v2 artifact keys and rule params are included by the existing public view. Old journals replay without these additions.

## Recompute and replicate

```
python -m sn120_v2.replicate rotation \
  --mining-json mining.json --round r000123 --work /data/affine
```

The command downloads the public decision and source IRT snapshot, reconstructs the previous state from public round evidence as described above, and compares the complete recomputed Rotation. `replicate run` uses the rotated round's active artifact and mirrored reserve for its catalog. No file under `sn120_v2/gpu/` changes.

## Enable (only with Jacob's approval)

1. Ingest and `verify` a reserve shard. Commit only the shard and manifest line; the live box reads it on the next round open after the normal code update.
2. Stop `run` between rounds and use normal `set-params` with a complete params file containing `rotation_enabled: true`. `rotation_k=32`, `rotation_retire_p=0.95`, `rotation_n_min=3`, low-retire off and `rotation_active_size=0` are the defaults.
3. Restart `run`. The first complete new round rotates before its draw. Existing open rounds are never changed in flight. Keep monitoring active size, retire/admit ids, queue and ledger; rollback is a normal between-round params change with `rotation_enabled: false`.
