# Continuous task rotation (off by default)

**Not merged, not deployed. Enabling needs Jacob's go.** Merging this code changes no round, pool, params hash, environment pin or GPU `code_digest`. Production activation is a separate `set-params` between rounds.

## Reserve layout

`sn120_v2/envs/reserve/` holds:

- `shards/<NNNN>-<slug>.jsonl`: immutable, canonical JSONL. Each row has `id`, `env`, `problem`, `answer`, `source` (dataset, revision and original row), `license`, `added_at` (the UTC `YYYY-MM-DD` ingest date) and `audit`. Ordinary rows have `audit.verdict == "keep"`. A global-prior row has exactly `{"method":"prior-v1 unscreened, prior_p from difficulty_meta","verdict":"unscreened"}` and may carry `prior_p`; it has passed the live grader answer-format self-check, not a difficulty or key screen. `math:` and `dapo:` prefixes are reserved for the base pool.
- `withdrawn.jsonl`: append-only `{id, reason, added_at}`. A withdrawn reserve task is not admitted; a withdrawn active task is retired outside the `k` cap.
- `MANIFEST.jsonl`: shard rows `{shard, sha256, count, added_at}` in admission order, followed by the current `{withdrawn: {sha256, count}}` pointer. `reserve_manifest_sha` is the SHA-256 of this file's bytes.

The shipped reserve has the original screened shards followed by the first global-prior batch. The loader checks every listed shard's bytes, canonical schema, unique ids and normalised problems. A later round must extend the shard history in the same positions and may only append withdrawals. Tampering fails closed with `ConfigError`; no round opens. `SN120_RESERVE_DIR` is a test-only override.

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

## Global-prior ingest (unscreened, tooling only)

The published [sn120-math-prior-v1 release](https://github.com/AffineFoundation/affine/releases/tag/sn120-math-prior-v1) is a candidate list, not a difficulty- or key-screened reserve. Prod's own evaluation is the screen. `ingest-prior` therefore labels these rows **unscreened**, without claiming they are audited `keep` rows:

```
uv pip install -e ".[envs,prior]"
python -m sn120_v2.envs.reserve ingest-prior \
  --prior-dir <MANIFEST-and-all-parquet-directory> --n 20000 --seed <seed> \
  [--min-p 0.15 --max-p 0.85 --no-signal-share 0.10]
python -m sn120_v2.envs.reserve verify
```

Before decoding rows it verifies every `MANIFEST.json` shard SHA-256, filename, schema and row count. `pyarrow` is an optional tooling dependency and is never imported by the controller. The release is not vendored.

The deterministic `prior_p` mapping is:

1. A published finite rate in `[0,1]`: `pass_rate_72b_tir` (also accepted under the published `pass_72b_tir` spelling), `llama8b_solve`, `medium_pass`, then generic `pass_rate` / `accuracy`, in that order.
2. Nemotron `gptoss_acc`: the unweighted mean of all its published gpt-oss rates. Every rate must be finite and in `[0,1]`.
3. OpenR1 `r1_correct`: a count out of eight, divided by eight.
4. A known difficulty label: Polaris `k/8` maps to `1-k/8`; `easy`, `medium`, `hard` map to 0.85, 0.5, 0.15. This is a deterministic estimate, not a new fitted model. Other numeric difficulty labels, such as DeepMath's `difficulty`, have no published pass-rate meaning here and get no prior.

Unsupported/missing metadata gets no `prior_p`; rotation then uses `rotation_prior_p`. Rows with `in_sn120_pool`, `key_from_majority` true (including the release's JSON string spelling), or an ID/problem already in the base, reserve, retired or withdrawn evidence are excluded before selection. The default band includes both endpoints: `0.15 <= prior_p <= 0.85`. At most `floor(n * no_signal_share)` no-signal rows are selected; this is a ceiling, not a quota, and defaults to 10%.

Candidates are ordered by `sha256(seed + "|" + prior_id)`, with source as a final deterministic guard. At most `0.40*n` selected rows may come from one source. Exactly N eligible rows are required; the command fails rather than silently returning a short batch. Every selected answer passes the same live `grade_answer` self-check and ordinary problem/answer/mandate bounds. One canonical JSONL shard is appended per source/license, in that order, with `env=affine_math`, `<source-prefix>:<next numeric id>` IDs, original dataset revision/row provenance and the exact unscreened audit block above. All earlier shards and their manifest rows stay byte-identical.

## Generator accepted tasks

`python -m sn120_v2.envs.reserve ingest-generator accepted-<epoch>.jsonl ... --prior-dir DIR` accepts generator drops whose rows have `problem`, `answer`, `source_id`, measured `prior_p`, `submitter`, `n` and `k`. The checksum-verified prior-v1 release supplies the original source and license; accepted rows use the same source-prefixed numeric task IDs as `ingest-prior`. The prior problem hash dedupes against the prior batch, base pool, reserve, retired and withdrawn rows, and each accepted answer must pass the live grader self-check. Accepted rows keep the measured `prior_p` and carry `{"method":"generator accepted, measured king pass rate","verdict":"screened"}`; they are not subject to `rotation_unscreened_max`. Public rows never contain submitter/n/k. Per-epoch provenance is written separately under `provenance/generator-<epoch>.jsonl` and is not read by the controller or used in replay. Re-running an accepted file adds no tasks. `python v2/ops/generator_ingest.py --prior-dir DIR [--dry-run]` processes pending epochs from fresh `origin/v2` worktrees; it records an epoch only after its reserve-only PR is opened and, unless `--no-merge` is set, its confirmed merge SHA is returned.

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

## Ranked active pool (`affine-pool-rotation-v2`)

Select this rule with `rotation_rule="info"`. The existing v1 rule and its bytes remain the default; selecting the new rule also needs `rotation_enabled: true`. The new default-omitted params are:

- `rotation_rule` (`"v1"` or `"info"`),
- `rotation_churn_frac` (default 0.10),
- `rotation_prior_p` (default 0.5),
- `rotation_queue_publish` (default 2000), and
- `rotation_churn_max` (default 0, no extra cap; when positive, an additional operational ceiling on steady-state swaps), and
- `rotation_unscreened_max` (default 0, **no cap**; when positive, the maximum number of `audit.verdict == "unscreened"` tasks admitted per round, counting fill and swaps together).

The candidates are the previous active set minus withdrawals, the unconsumed reserve queue, and the non-withdrawn retired archive. Withdrawn tasks never return. A v2 retirement is not permanent: a task can become a high-information candidate again. A returning task restores its exact archived `(difficulty, information, rounds_seen)` row and its head-ledger evidence; a brand-new task uses the same head-ability prior as v1. The IRT table keys remain exactly the active set.

For each candidate, score `p * (1-p)`, the Fisher information of a Bernoulli outcome. An active or archived task with `rounds_seen >= rotation_n_min` uses `predicted_pass(head_ability, difficulty)`. Otherwise use the public reserve row's `prior_p` if present, else `rotation_prior_p`. **Saturated tasks keep their score**, even when `p > rotation_retire_p`; `high` is their retirement reason. Only a **dead** task (`head_rollouts >= rotation_n_min_low` and `head_passes == 0`) is forced to score zero. `low` is its retirement reason.

With `T = rotation_active_size or the initial active size`, apply this order every round:

1. Remove withdrawn active tasks, outside the churn cap.
2. If above target, retire the lowest-scored `|A|-T` active tasks, with no cap. Reasons are `excess`, or `high`/`low` for flagged tasks.
3. If below target, admit the highest-scored outside candidates, with no cap on screened candidates. A positive `rotation_unscreened_max` caps raw admissions across this fill and step 4 together; skipped raw candidates stay in the outside queue for later rounds.
4. While a strictly higher-scored outside candidate exists, swap it for the lowest-scored active task, at most `floor(rotation_churn_frac * T)` times (and additionally at most `rotation_churn_max` when that is positive). Reasons are `high`, `low`, or `displaced`. Thus the swap count is reserve-bounded: it never exceeds the number of outside candidates strictly better than the active task replaced, and it is zero when no such candidate exists. Shrink/fill and withdrawals are not swaps and are not capped by these settings.

Retirements are ascending by score; admissions are descending. Every tie uses `sha256(f"{round_id}|rank|{task_id}")`, with task ID as the final deterministic guard. There is no random number, wall-clock time, Python `hash()`, or dependence on shard/input order. `next_up` is the top `rotation_queue_publish` remaining outside candidates in rank order, not FIFO order. Reserve metadata is read once at runtime construction; the per-round selection is `O(|C| log |C|)` and carries only IDs, priors, IRT rows and ledger counts.

### Public v2 files and replay

In addition to the existing active, rotation, state, reserve and retired files:

- `public/pool/ranking/<round_id>.json` contains every candidate while the candidate count is at most 100,000: `id`, previous `state`, `p`, `rounds_seen`, `score`, `flags`, `source_prefix`, `env`, `head_rollouts`, `head_passes`, `unscreened` when true, and `admitted`/`retired`/`retire_reason` for this decision. Above 100,000 candidates it publishes all active and all retired-state rows, plus the top 20,000 remaining outside candidates in rank order. The file and Rotation carry `ranking_complete: false` for this bounded view. It never truncates replay inputs: the full queue, reserve history, IRT table, ledgers and retired archive remain in their existing public artifacts, and admitted IDs are derived from the complete decision rather than the view.
- `public/pool/rotation/<round_id>.json` carries the same complete candidate ranking plus admitted IDs, retired IDs/reasons, `next_up`, `rule_params` and `retire_reason_counts`.
- `public/pool/summary.json` aggregates the post-decision active composition and score histogram, reserve/retired counts, and the last 20 v2 rounds for the static `/pool` page (`pool.html`). The page has a top-50 upcoming queue and a client-side search over the complete latest ranking. All times are UTC.

The first v1-to-v2 transition publishes the complete previous retired archive, candidate metadata and retired ledger it used. `python -m sn120_v2.replicate rotation --mining-json <mining.json> --round rNNNNNN` fetches the public decision, ranking, source snapshot and state lineage and compares the complete recomputed `Rotation`, including the full ranking. The early live v1 `r000066` state is byte-SHA addressed; the later `r000069` state uses the canonical digest. The loader verifies the exact public key's SHA-256 and permits that legacy v1 state at the transition, while new v2 states always use the canonical digest.

Strict replayer schema additions (all optional and omitted on v1/old journals):

- `Params`: `rotation_rule`, `rotation_churn_frac`, `rotation_prior_p`, `rotation_queue_publish`, `rotation_churn_max`, `rotation_unscreened_max`.
- `Reserve`: `audit_verdicts` (sparse non-`keep` verdicts; retired raw rows retain `audit_verdict` in their archive rows).
- `Rotation`: `ranking_complete` (omitted when true; false for the bounded large-pool view). Ranked candidate rows include optional `unscreened: true`.
- `Reserve`: `candidate_meta` (per-ID `source_prefix`, `env`, optional `prior_p`).
- `Rotation`: `candidate_ranking`, `retire_reason_counts`, `restored_irt_rows`, `initial_retired_archive`, `initial_candidate_meta`, `initial_retired_ledger`, `retired_ledger`, `opened_at`.
- Journal: no new record kind and no new `RoundRecord` field. The existing optional `RoundRecord.rotation` map additionally carries `retire_reason_counts` and the v2 `rule_version`; v2 artifact keys and rule params are included by the existing public view. Old journals replay without these additions.

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
