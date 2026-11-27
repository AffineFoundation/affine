# PART 0 — withdrawal dry run against the current production code

**Verdict: PASS.** Withdrawing the 300 ids in `/home/bb/gq-guard/withdraw_batch1.jsonl` is safe to do on
the prod box with the code at `8abc4ca5`, with one operational condition (§6) and one capacity caveat (§7).

No repository file was modified for this exercise. Everything below was produced by running the
unmodified tree at `8abc4ca5` against a reserve directory rebuilt from the public files and a scratch
copy of the public mirror. Production was not touched: nothing was written to `/root/sn120-run`, no
journal lock was taken, no object was uploaded.

All times UTC.

## 1. What the live state was

Read from `https://affine.io/public/...` with a browser User-Agent (as `envs/reserve/generator.py::_active_ids` does).

| | |
|---|---|
| latest round | `r000160` (open — `public/rounds/r000160/round.json` is 404, as expected) |
| previous closed round | `r000159` |
| rule | `affine-pool-rotation-v2`, `rotation_rule = "info"` |
| `rotation_active_size` | 10000 |
| `rotation_churn_frac` / `rotation_churn_max` | 0.1 / 0 |
| reserve | 144 shards, `withdrawn.jsonl` **empty** |
| retired archive | 26,426 rows |
| `queue_remaining` of r000160 | **0** |

The 300 ids: all unique; **298** are in r000160's active set, 2 (`gen:6156`, `gen:6086`) are already in the
retired archive; **299** were active at r000159 (the set the replayed open rotates).
Prefixes: `gen` 273, `dapo` 19, `crys` 3, `dmath` 3, `nmv2` 2.

## 2. The withdrawals were applied with the real CLI

```
SN120_RESERVE_DIR=/home/bb/gq-part0/reserve \
  python3 -m sn120_v2.envs.reserve withdraw <id> --reason <reason>
```

(`__main__.py`: positional `id`, required `--reason`; `_withdraw_locked` appends
`{"id","reason","added_at"}` to `withdrawn.jsonl`, rewrites the manifest pointer line and then calls
`load_reserve(root)` to verify.)

**300 applied, 0 failures**, ~5 s each (each call re-verifies the whole reserve). Reasons are all
`quality:<category>` as supplied in the batch file.

## 3. Baseline: the harness reproduces the published round byte-identically

Replaying r000160's open from the **unmodified** reserve, through
`PoolRuntime.__init__ → open() → rotation.rotate_info`:

```
active_sha a57f67fa771b6524373704f98be73b7ac0ebd889780e3598e73b92273194b907   (== published)
active_size 10000   admitted 75   retired 75
```

So the harness is a faithful stand-in for the loop; the deltas in §4 are caused by the withdrawals and
nothing else.

## 4. The next open with the 300 withdrawals applied

Sequence matches production: the `PoolRuntime` is constructed **before** the withdrawals land, so the
reserve reload at open (`_publication → ingest_lock(exclusive) → _reload_at_open → extends_reserve`) is
the code under test.

```
reserve reloaded at open: 144 shards, 9909 queued,
  085c6d9d55c15017f06f0d43a0f0e810018500bac64299be5d35cb7424fa3c0d
```

(10190 → 9909 = the 281 withdrawn ids that are reserve rows; the 19 `dapo:` ids are base-pool rows.)

| assertion | result |
|---|---|
| open raised no exception | **yes** |
| reserve reloaded at open | **yes** |
| active size still 10,000 | **yes** |
| withdrawn active ids retire with reason `withdrawn` | **yes** — 299/299 |
| no withdrawn id remains active | **yes** |
| no withdrawn id appears in `next_up` | **yes** |
| catalog contains every new active id | **yes** (0 missing) |
| admitted ⊆ active, disjoint from the previous active set | **yes** |
| deterministic (identical `Rotation` record on a clean re-run) | **yes** |
| `verify_public_pool_snapshot` | **PASS** |

Resulting round:

```
active_sha be961db9f8a6f92a925c91e61ee974701383fe444b12727cacac8eda3e6ec790
active_size 10000   admitted 324   retired 324
retire_reason_counts {"withdrawn": 299, "displaced": 23, "low": 2}
next_up 2000        queue_remaining 0
```

324 = the usual 75 churn plus 249 extra admissions to refill the hole the withdrawals left. The pool
self-heals in a single round.

## 5. Loop restart with local withdrawals ahead of the public mirror — accepted

Constructed a `PoolRuntime` with a local reserve carrying the 300 withdrawals against a mirror whose
`withdrawn.jsonl` is still empty: **all 300 are kept** (`withdrawn_count == 300`).

Why, in `pool_runtime._restore_public_cache` (lines 417-482): the comparison excludes the trailing
pointer line (`here[:-1]` / `remote[:-1]`), so the shard prefix compares `equal`, `restore` stays
`False`, and the local `withdrawn.jsonl` is never overwritten. The only failure mode is `older` —
local having **fewer shards** than the mirror while also carrying extra withdrawals — which raises
`VerifyError("local reserve withdrawals do not extend its public mirror")`. On prod the local reserve
is the source of the mirror, so it cannot be behind.

## 6. Mid-round withdrawal: safe while the loop runs, **not** safe across a mid-round restart

Two cases, both measured:

**A. loop keeps running (the normal case).** The catalog is the object built at the last open. Dropping
the new `withdrawn.jsonl`/`MANIFEST.jsonl` into the reserve dir changes nothing in the live process:
all 300 ids still resolve via `pool.task(...)` afterwards (`resolvable_before == resolvable_after == 300`).
Nothing re-reads the reserve until `_reload_at_open` at the next open. The round in progress closes
normally.

**B. loop restarts mid-round.** `PoolRuntime.__init__` reloads the reserve, and **281 of the 300 ids
vanish from the catalog** (`ContractError: task ... is not in the math pool`). The 19 `dapo:` ids
survive because `Catalog.__init__` seeds `by_id` from the whole base pool and `_load_reserve` only pops
reserve rows. **279 of the 281 are in the open round's active set.**

Every catalog/task lookup between open and close, and what each does with a missing id:

| path | lookup | effect |
|---|---|---|
| `v2/sn120_v2/gpu/evaluator.py:469` | `self.pool.task(task).messages()` — builds the eval request for a drawn task | **raises `ContractError`** |
| `v2/sn120_v2/gpu/evaluator.py:950` | `self.pool.task(tasks[job[2]])` — grading a finished job | **raises `ContractError`** |
| `v2/sn120_v2/rollouts.py:114` | `self.pool.task(request.task_ids[index])` — gate rollout capture | **raises `ContractError`** |
| `v2/sn120_v2/admission.py:204` | `task_for_prompt` | degrades to `None`, no raise |
| `v2/sn120_v2/public.py:598` | `task_for_prompt` for batch source stats | degrades to `"unmapped"` |
| `v2/sn120_v2/controller.py:593` | best-effort lookup | already guarded |
| `v2/sn120_v2/pool_runtime.py:314`, `:639` | archive publication | `try/except`-wrapped |
| `rotation.fold_ledger` | ids only | safe |
| IRT snapshot | ids only | safe |

So a withdrawal applied mid-round is invisible until the next open, but a crash/restart before that open
turns the still-active withdrawn ids into hard `ContractError`s on the three grading paths.

**This is exactly why PART 3's `apply` must run between opens** — after `open: the public files of round
<id>` for the current round, so the withdrawal is picked up by the very next `_reload_at_open` and the
window in which a restart could hit it is as short as possible. It is also why `apply` must cap at 300
and never run twice in the same round.

## 7. Capacity caveat for the operator

Production's `queue_remaining` is already **0** — the reserve queue is exhausted, and every admission
now comes from the **retired archive** (26,426 rows), which `rotate_info` includes in `outside`. The
refill in §4 (249 extra admissions) came entirely from there.

Consequences for PART 2/3:
- the fail-closed gate's eligible-candidate supply is the *retired archive*, not the reserve queue, so
  the pre-judge worker must judge `next_up` (which is drawn from the archive) and not wait for new
  reserve ingest;
- `apply`'s "refuse to withdraw more active ids than (eligible pass candidates outside − 50)" buffer is
  the binding constraint from day one under `info-guard`; with an empty guard ledger it will refuse
  every withdrawal, which is the intended fail-closed behaviour — the ledger has to be populated first.

Under the current `info` rule (what prod runs today) there is no such gate, so **batch 1 of 300 can be
withdrawn now**, before any of PARTs 1-3 land.

## 8. How to reproduce

Scripts live outside the repo in `/home/bb/gq-part0/scripts/` (they are a verification harness, not
product code): `fetch_public.py`, `fetch_states.py` (walks `prev_state_sha` back 92 states to r000066),
`build_reserve.py`, `replay_open.py`, `midround.py`. Machine-readable result:
`/home/bb/gq-part0/work/replay_result.json`.
