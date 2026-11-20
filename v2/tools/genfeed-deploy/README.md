# Generator-feed deploy kit (operator only)

This directory deploys the exact consumer commit named in `expect.json`, from **28c56f439c0c0fbf3554c8546cc86dbd88119125**. The kit is included in the following kit-only PR commit; `new_commit` intentionally names the deployed consumer tree, not that later metadata-only head. It is an operator kit, not an automatic deployment. Development verification is `dry.sh` against the supplied r080 copy and a scratch clone/public mirror. No production, R2 or secret access is used.

The consumer accepts `generator-intake-v2` rows with `king` as well as the four-field format and the existing `prior_king`, `king_revision`, `source`, `license`, and `task_id` fields. Revision precedence is `king`, `prior_king`, `king_revision`, `results.king`, then `unpublished`; every supplied row revision must be a lowercase 64-hex SHA-256. A row-derived revision is retained in the audit ref as `#king=<revision>`. The original failed consumer rejected all 885 0400Z rows as `schema`; the corrected intake keeps 318.

## Production prerequisites

- The live checkout is at `old_commit`, with the supplied r080 journal and untracked `retired.json` / `retired_tasks.json` beside the packaged reserve.
- `/root/sn120-run/generator/inbox/accepted-EPOCH.jsonl` and, when available, `accepted-EPOCH.done` (`generator-intake-v2`: `epoch`, `lines`, `sha256`, `format`) or public `results.json`.
- The existing saved environment is unchanged. `sn120-v2-run-loop.sh` reads `/root/sn120-run/reserve-live.path` itself. The operator must not add an override to the saved environ file.
- `/root/sn120-run/generator/prior-index.json` must be the exact verified artifact in `expect.json`, or `apply` refuses.
- `pyarrow` is needed only for `stage`, not for `apply` or ingestion. If the production venv lacks it, build the index on another host with NEW's builder and place the exact artifact below; `apply` verifies it and reads JSON only.

## Stage (safe while the loop runs)

From the kit directory:

```bash
python3 ol_operator.py stage
```

This downloads the public [AffineFoundation/affine `sn120-math-prior-v1` release](https://github.com/AffineFoundation/affine/releases/tag/sn120-math-prior-v1), verifies the pinned manifest SHA and all eight parquet SHAs, extracts `new_commit` with `git archive` into a scratch directory, builds with `sn120_v2.envs.tools.prior_index`, verifies the resulting index, and atomically installs `/root/sn120-run/generator/prior-index.json`. It never uses the live checkout for the build. The default kit location, release cache and build scratch are all under `/root/sn120-run/generator/`.

Expected artifact:

- schema/pool: `sn120-math-prior-index-v1`, `sn120-math-prior-v1:45b793b4d00d1f7225d34fd35e58018612a0ded00d96a9c23b750cfb18994115`
- rows/source_rows: **811,609 / 811,609**
- bytes: **111,176,289**
- SHA-256: **`04301d0afac965907b8c03a7af6fe1254c0bd646f449cd14fee564439307da25`**

## Apply / run

```bash
GATE_WANT=r000080 bash autodeploy.sh
```

`autodeploy.sh` waits for open r000080 on old HEAD, touches stop-after-round, waits for STOPPED/loop-gone, verifies the journal close, and DISARMs the watchdog. `$H/disarm` before apply aborts without changing code. `apply` verifies HEAD, unchanged code digest and params, pool, packaged reserve and the exact prior index; writes the external selector; records the rollback target; and seeds the live reserve.

Before seeding, `apply` checks that LIVE is the exact packaged seed: matching manifest and withdrawal hashes, exactly the listed shards, and no other state. A Generator ledger, retired files, extra shard/reject files, a different manifest, or any other non-clean state causes the whole LIVE directory to be renamed to a timestamped sibling `reserve-live.backup-*`; a fresh packaged reserve is then seeded. The backup is never deleted by the kit, and the packaged in-tree reserve is never used as LIVE or written by seeding.

`ingest-ready` ingests **only the oldest complete pending epoch** and skips later/future epochs for the daemon. For the supplied inbox, that is 0400Z (**318** kept); 0800Z is deliberately not ingested yet. If no complete epoch is ingested, or the just-ingested epoch kept zero tasks, it exits nonzero and never prints `INGEST READY DONE`; `autodeploy.sh` therefore rolls back before starting the loop. Exactly one run loop starts. The first open health check must pass **before** the detached ingestd starts (`setsid nohup`, `/root/sn120-run/generator/ingestd.pid`). Ingestd then brings in later epochs for following rounds. No daemon runs during apply or the first open.

The first open must have params `94f931588114f335814231409938f39f2056893708bc798e90f1f7676920a35c`, size 10,000 and at least one admitted `gen:*`; otherwise the script stops after the round and rolls back: stop ingestd, remove `reserve-live.path`, checkout old_commit, start the loop. It polls the public reserve manifest, Generator shards and rotation for 20 minutes, then requires a clean journal close and the next open in the same loop process before PASS and watchdog re-arm.

The clean-packaged-prefix replay of this exact ingest/open sequence, starting from the supplied r080 journal and public state and closing each prior round before the next ingest/open, measured:

| round | epoch ingested before open | total admitted | `gen:*` admitted | retired | active size |
| --- | --- | ---: | ---: | ---: | ---: |
| r000081 | 0400Z before start | **318** | **318** | **318** | **10,000** |
| r000082 | 0800Z after first-open PASS | **554** | **554** | **554** | **10,000** |
| r000083 | 1200Z | **408** | **408** | **408** | **10,000** |
| r000084 | 1600Z | **430** | **430** | **430** | **10,000** |

All four rotations passed append-only and replay verification, with the unchanged params hash. These are measured backlog admissions from the supplied inputs and state, not a guaranteed count if active/king evidence or inbox contents differ.

`one-link-ol_operator.py` is the operator's existing helper, included for reference; the new operator includes the required journal-close check directly. `prod-run-loop.reference.sh` is the old production run-loop reference; the new checkout supplies the selector-enabled script.

## Paths

```text
kit:      /root/sn120-run/generator/genfeed-deploy/
selector: /root/sn120-run/reserve-live.path
contents: /root/sn120-run/reserve-live
index:    /root/sn120-run/generator/prior-index.json
inbox:    /root/sn120-run/generator/inbox
ingestd:  /root/sn120-run/generator/ingestd.log
pidfile:  /root/sn120-run/generator/ingestd.pid
```

The loop exports `SN120_RESERVE_DIR` from the selector only when it exists; otherwise it leaves the saved environment untouched. `ingestd` reads the same selector. Public results default to `https://affine.io`; `SN120_PUBLIC_URL` can override that for the daemon. No params change is needed: churn stays capped at 1,000.

## Dry run

`dry.sh` clones to a new scratch directory, checks out old_commit, copies the supplied r080 journal/retired files, verifies the exact pinned prior-index artifact with the kit's byte/SHA/schema checks, and serves an isolated local public mirror. It is the apply/ingest harness, not a `stage` rebuild test; the pinned index was separately built from the hash-verified release assets. It verifies missing/wrong-index refusal and the open-round journal gate, prepares the failed deployment's dirty LIVE, then performs one apply and verifies its timestamped backup and fresh packaged seed. It requires the supplied live-v2 `SHA256SUMS`, asserts only 0400Z is ingested before start-loop, and separately checks that no-complete and zero-kept ingestion fail closed. The measured four-round replay supplies the health/public gates for r000081–r000084; later epochs are ingested only after the first-open gate. It checks public Generator shards/rotations and rolls back to the exact old commit with the selector removed. Neither a production loop nor a daemon is started. Its log is `genfeed-dry/dry.log` below the scratch parent chosen by `D`.

## Supply / admission planning

The four supplied live-v2 epochs contain **3,328 rows** and keep **1,710 tasks** in sequential ingestion: **318, 554, 408, and 430**. There are **zero schema rejects**. The replay admits one epoch per round in the measured **318 → 554 → 408 → 430** sequence rather than admitting 1,710 together. The real-input checksums, pinned prior index, clean packaged reserve, and synthetic scored-null closes make this an offline estimate of the supplied backlog, not a claim about future Generator supply.

No params or rotation change is needed. The 10% rule still caps total churn at **1,000**; actual Generator admissions depend on current king informativeness and dedupe.
