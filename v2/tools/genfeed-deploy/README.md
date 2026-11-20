# Generator-feed deploy kit (operator only)

This directory deploys the exact consumer commit named in `expect.json`, from **28c56f439c0c0fbf3554c8546cc86dbd88119125**. The kit is included in the following kit-only PR commit; `new_commit` intentionally names the deployed consumer tree, not that later metadata-only head. It is an operator kit, not an automatic deployment. Development verification is `dry.sh` against the supplied r080 copy and a scratch clone/public mirror. No production, R2 or secret access is used.

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

`ingest-ready` ingests **only the oldest complete pending epoch** and skips later/future epochs for the daemon. For the supplied inbox, that is 0400Z (**318** kept); 0800Z is deliberately not ingested yet. Exactly one run loop starts. The first open health check must pass **before** the detached ingestd starts (`setsid nohup`, `/root/sn120-run/generator/ingestd.pid`). Ingestd then brings in 0800Z for the following round. No daemon runs during apply or the first open.

The first open must have params `94f931588114f335814231409938f39f2056893708bc798e90f1f7676920a35c`, size 10,000 and at least one admitted `gen:*`; otherwise the script stops after the round and rolls back: stop ingestd, remove `reserve-live.path`, checkout old_commit, start the loop. It polls the public reserve manifest, Generator shards and rotation for 20 minutes, then requires a clean journal close and the next open in the same loop process before PASS and watchdog re-arm.

The replay of this exact ingest sequence measured:

| round | total admitted | `gen:*` admitted | retired | active size |
| --- | ---: | ---: | ---: | ---: |
| r000081, 0400Z before start | **318** | **318** | **318** | **10,000** |
| r000082, 0800Z after first-open PASS | **554** | **554** | **554** | **10,000** |

Both rotations passed append-only and replay verification. These are the measured backlog admissions, not a guaranteed count if king evidence or inbox contents differ.

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

`dry.sh` clones to a new scratch directory, checks out old_commit, copies the supplied r080 journal/retired files, rebuilds the prior index from hash-verified release assets using NEW's archived builder, and serves an isolated local public mirror. It verifies missing-index refusal and the open-round journal gate, applies the deploy, and asserts only 0400Z is ingested before start-loop. The measured 318-admission r000081 replay is the health gate; only then does the dry run ingest 0800Z and check the measured 554-admission r000082 replay. It checks public Generator shards/rotations and rolls back to the exact old commit with the selector removed. Neither a production loop nor a daemon is started. Its log is `genfeed-dry/dry.log` below the scratch parent chosen by `D`.

## Supply / admission planning

The real 0400Z and 0800Z epochs kept 318 and 554 tasks: 872 over 8 hours, or 436 per 4 hours at the same rate (**109/hour**). At one round every 1.5–1.9 hours, this is about **164–207 new Generator tasks per round** before further dedupe against the growing catalog. An entire four-hour epoch can be larger: the observed 0800Z epoch is 554. The deploy backlog is intentionally split into the measured **318 then 554** sequence rather than admitting 872 together.

No params or rotation change is needed. The 10% rule still caps total churn at **1,000**; actual Generator admissions depend on current king informativeness and dedupe.
