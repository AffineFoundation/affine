# The benchmark service of SN120 v2: operator steps

This file is for the operator. It says which host runs the service, which GPUs it uses, how to start and stop it, what you see when it is right, and the way back.

Words used here:

- **Row**: one model in the benchmark table. A row is an official public reference, the launch model (the genesis head), or a king.
- **King**: the model after a round that a miner won. The row is the folded head: the base with the winner's LoRA merged in.
- **Node**: the GPU machine that the service rents on Lium for the time of a run.
- **Store**: the R2 bucket of v2.

## 1. What the service does

- It reads three kinds of public files from the store: `public/head.json`, `public/heads/<id>.json` and `public/rounds/<round>/round.json`. It calls nothing on the GPU box.
- From them it makes one row for the launch model and one row for each king. `bench/config/references.toml` adds the official public reference row, Qwen3.5-9B.
- A row with a missing result is queued in this order: official references, genesis, then kings newest first. One row runs at a time.
- For a queued row the service rents one node and runs the 6 benchmarks of `bench/config/benchmarks.toml`. Genesis and kings load from the store with expiring read links. The official reference uses a public, pinned Hugging Face snapshot with no write token or store credential on the node.
- It writes one object: `public/benchmarks.json`. It writes it on each change of a row or of a result. It deletes nothing in the store.
- The public pull on AffineValidator copies that object to the site, as it does today for the other public files. The site builder already reads it.
- The table is for information only. No code that picks a winner or sets weights reads it.

A row is run once. The service reads the published table before its first step. A result that is in the table is done, also on a host with an empty disk. Publishing is merge-only: retired benchmark scores and v1 reference history are not deleted.

### Option A public math columns (25 Nov 2026 UTC)

The six enabled and counted columns are HMMT Feb 2025, HMMT Nov 2025, HMMT Feb 2026, BRUMO 2025, CMIMC 2025 and
BeyondAIME. Every one uses 8 rollouts and a pinned public dataset. Avg is their unweighted mean; no pooled-all-six
column is counted. A row missing any one of the six has no Avg.

The saturated and older columns are disabled, removed from Avg and marked `hide_column = true`. Their registry entries,
store results and row cells remain. The home table hides those columns; the v1 archive is unchanged. The official v2
Qwen3.5-9B row supersedes only its duplicate frozen v1 row on the home table after the v2 row exists.

Deploying the new registry queues the missing six-benchmark runs for the official reference, genesis and each king, in
that priority order (kings newest first). It does not rent until the existing daemon is updated and processes the queue.
There is still only one daemon, one pod and one node run. To roll the site change back, revert the benchmark registry
and site-builder changes; already-published new-benchmark results remain in the store because publication is merge-only.

### GSM8K-Platinum history (18 Nov 2026)

The original GSM8K test set had approximately 110 flawed or ambiguous questions and 10 wrong answer keys.
**GSM8K-Platinum** (`madrylab/gsm8k-platinum`, revision
`e762492455a1cf7967de89f05b6bef72fc713b66`, `main/test`) replaced it with the cleaned 1,209-question set. Option A
subsequently retired both GSM8K columns: they are disabled, hidden and outside Avg. Their published scores remain in
the store and row cells. The root-level v1 `bench/` is unchanged.

The operator's AffineValidator checkout may have local edits to `v2/site/build.py`. Merge the site-builder change by
hand there or leave it out by the operator's decision; do not discard those local edits. Without the site hunk, retired
columns and the official-v2 Qwen display rule are not applied.

For past kings, expect scores of approximately **0.92–0.96 on original GSM8K to become 0.96–0.98 on Platinum**. This is an
expected range, not a guarantee for every model.

**The Platinum backfill was superseded by Option A. Deploying the current registry does not queue Platinum or any other
retired benchmark.** The 18 Nov 2026 plan had 23 existing launch-model and king rows needing a `gsm8k_platinum` run. Each
would have been a separate per-row evaluation on the single rented node, obtaining and verifying that row's weights and
starting vLLM. Each new rent would repeat installation and weight download, subject to `BENCH_MAX_RENTS_PER_DAY` (8 rents in
24 hours by default). That historical backfill is not billed and is no longer scheduled. The current Option A queue and
deployment decision are described above.

For historical estimation from saved private per-task files:

```
python -m affine_bench.backfill_gsm8k_platinum "$BDIR/state/private" --out gsm8k-platinum-estimates.json
```

This reads each `<row>/gsm8k.per_task.jsonl`, maps questions by exact text, and reports the old score, the score over the
1,199 unchanged questions, and lower/upper bounds over all 1,209 with the 10 revised answers counted wrong/right. It fetches
the two datasets from Hugging Face by default; `--original-jsonl` and `--platinum-jsonl` accept local dataset exports instead.
It prints or writes JSON only. It never writes to the store or substitutes an estimate for a real benchmark result.

## 2. Which host

| Host | What this change needs there |
| --- | --- |
| **Dublin** (the GPU box) | Nothing. No checkout, no restart, no new process. The code digest does not change. |
| **AffineValidator** | The service runs here by default: one process with no GPU. |

Why AffineValidator is the default:

- The service needs a CPU, the checkout, the Lium command, and a store token. It needs no GPU of its host.
- The pull that puts the table on the site runs on this host already.
- v1's benchmark service (`affine-bench.service`) is in the runbook's list of v1 units for this host.

Cost of this choice: the host gets one more credentials file, and its token can write to the bucket. The service itself writes one key only (`public/benchmarks.json`).

Another CPU host works the same way. It needs the four things of section 4.

Dublin as the host of the process is not the default. The process uses no GPU, but it would be a new process on the box that runs the round. Its load there is not measured.

## 3. Which GPUs

| Choice | What it costs | Can it change a round's result? |
| --- | --- | --- |
| **A. A node that the service rents for each run (the default)** | The price of the node for the time of the run, the setup and up to 900 seconds of idle time. Never above `BENCH_MAX_PRICE` dollars per hour. At most 8 rents in 24 hours. | No. It is another machine. It shares no GPU, no disk and no process with the round. |
| **B. Dublin's GPUs** | No rent. No round while a benchmark runs. | Yes. Not built, and not to be done by hand. |
| **C. A node that stays rented** | The price of the node for every hour, also with an empty queue. | No. |

Details of A:

- The service asks Lium for a whole node of the secure tier, in this order: 8x B300, 8x B200, 8x H200, 8x H100 (`BENCH_GPU_PREF`, default `B300:8 B200:8 H200:8 H100:8`). It takes the cheapest free node of the first kind that has one at or below the cap.
- The node has the name `affine-v2-bench-1`. The service rents no second node while Lium lists a pod of that name.
- It gives the node back 900 seconds after the queue is empty (`BENCH_IDLE_RELEASE_S`). It removes only the pod with the exact Lium id that it rented.
- Each new rent installs the software again and loads the weights again. The launch model's weights are 18.85 GB (the sum in `public/read_links.json`). The public Qwen reference has no store credential; its snapshot download uses `token=False`.
- The time and the price of one row on v2 are not measured. v1's figures, from v1's public file `https://affine.io/benchmarks.json` on 14 Nov 2026: a median of 3582 seconds for one model (`progress.median_model_wall_s`), on a node of 8x B300 at 32.0 dollars per hour (`node`). v1 ran the same 10 benchmarks with the same harness.

Details of B:

- A round uses all 8 GPUs (`n_gpus = 8`). The benchmark's servers take 0.90 of each GPU's memory (`bench/config/serving.toml`).
- The round's evaluator starts only when no GPU holds more than 1024 MiB. It waits 120 seconds, then it stops with an error (`sn120_v2/gpu/evaluator.py`). So a round that meets a benchmark cannot score.
- Also without a clash, the time of a benchmark is time with no round. A round has at most 100 minutes (`round_time_limit_s = 6000`).
- The service has no setting for this choice. It always rents.

Details of C: set `BENCH_IDLE_RELEASE_S` to a large number of seconds in the settings file. The node then stays after a run. Use this only if kings come so often that the setup of each rent costs more than the idle hours.

## 4. Before the first start

Do these on the host of the service, as the user that will run it.

```
REPO=<the checkout of the repository on this host>
PY=<the python of the virtual environment of v2, as a full path>
RUNUSER=$(id -un)
LIUM=$(command -v lium)
BDIR=$HOME/sn120-v2-bench
BENV=$BDIR/secrets/r2-bench.env
BSET=$BDIR/settings.env
mkdir -p "$BDIR/state" "$BDIR/site" "$BDIR/secrets" && chmod 700 "$BDIR/secrets"
```

1. **The code.** The checkout is at the commit that has this file, or a later commit of `v2`. Install the service into the environment of v2:

   ```
   cd "$REPO/v2" && uv pip install --python "$PY" -e ".[r2]" -e ./bench
   git -C "$REPO" branch -r --contains HEAD     # must list origin/v2: the node takes this commit from GitHub
   ```

2. **The Lium command.** `"$LIUM" ps` lists the pods of the account and asks for nothing. The Lium key is in this user's Lium configuration, and Lium knows an ssh key of this user. No line of the list has the name `affine-v2-bench-1`.

3. **The store token.** `$BENV` has mode 600 and four lines: `SN120_R2_ENDPOINT`, `SN120_R2_BUCKET`, `SN120_R2_ACCESS_KEY_ID`, `SN120_R2_SECRET_ACCESS_KEY`, with the production values of the runbook. No `SN120_R2_PREFIX`. The token must read and write objects in the bucket. The read-only token of the public pull is not enough. Which token is used is Jacob's choice.

4. **The settings.** They hold no secret.

   ```
   cat > "$BSET" <<EOF
   BENCH_STORE=sn120_v2.r2:R2Store.from_env
   BENCH_STATE=$BDIR/state
   BENCH_SITE=$BDIR/site
   BENCH_NODE_SSH=$BDIR/node-ssh
   BENCH_LIUM=$LIUM
   BENCH_REPO_REF=$(git -C "$REPO" rev-parse HEAD)
   BENCH_MAX_PRICE=<the highest price in dollars per hour that Jacob allows>
   EOF
   bench() { ( set -a; . "$BENV"; . "$BSET"; set +a; "$PY" -m affine_bench.cli "$@" ); }
   ```

   A missing setting stops the service at its start with the name of the setting. `bench` is the command of the next steps.

5. **The queue, with no cost.** `status` only reads.

   ```
   bench status
   ```

   The output has the official `ref-qwen3.5-9b` reference first under `next`, then `genesis-87de8304e9d33dde`, then any kings with missing enabled results. `"pod": null` while no pod is rented. A store error here means that the token cannot read.

6. **The official reference needs no seed or HF token.** `bench/config/references.toml` supplies its public model id, pinned revision and profile. The service builds that row itself and downloads the public snapshot on the node. The site's builder (`v2/site/build.py`) takes the other reference models from the frozen v1 history on AffineValidator, on each build, whatever this table holds. So start with an empty `$BDIR/site`: copy no file into it.

   This holds only when AffineValidator runs the site builder that has this rule. Before the next step, on AffineValidator (`REPO` and `SITE` as in `v2/site/OPERATIONS.md`):

   ```
   grep -c '^def tried' "$REPO/v2/site/build.py"
   ls -l "$SITE/public/benchmarks.json"
   ```

   - Right: the first command prints `1`, and the second prints `No such file or directory`.
   - The first command prints `0`: the checkout is older. Stop here and do the steps of `v2/site/OPERATIONS.md` first. With the older builder the site shows only the rows of the store's table, so the reference models leave the site at the next step.
   - The file is there: it is not the service's file, because the service has not written yet. Do step 6 of `v2/site/OPERATIONS.md` (it moves the file out of the site directory). While the file is there, the pull does not copy the table of the store.

7. **The first write, with no rent.** `publish` writes the table and rents nothing.

   ```
   bench publish
   ```

   It prints a line with `"event": "table-read"`, a line with `"event": "table-written"` whose `rows` has `"ref-qwen3.5-9b": "queued"`, `"genesis-87de8304e9d33dde": "queued"`, and one entry for each other id that step 5 listed, then `published`. A store error here means that the token cannot write.

   After the pull's next run (one minute): `curl -s -o /dev/null -w '%{http_code}\n' https://affine.io/public/benchmarks.json` prints `200`. The site's table has the same reference models as before, and its row `Genesis` has the state `queued` in place of `no run yet`.

   If the pull's log says `public/benchmarks.json is in the site directory and this command did not write it`, a file placed by hand is in its way (step 6): move it out of the site directory. The next run of the pull copies the table of the store.

## 5. Start

The start rents a node as soon as a row waits. After section 4 that is at once.

```
sed -e "/^[^#]/s|SN120_USER|$RUNUSER|" -e "/^[^#]/s|SN120_ENV_FILE|$BENV|" \
    -e "/^[^#]/s|SN120_BENCH_SETTINGS|$BSET|" -e "/^[^#]/s|SN120_PYTHON|$PY|" \
    "$REPO/v2/bench/deploy/sn120-v2-bench.service" | sudo tee /etc/systemd/system/sn120-v2-bench.service > /dev/null
grep -v '^#' /etc/systemd/system/sn120-v2-bench.service | grep SN120_      # must print nothing: every blank is filled
systemd-analyze verify /etc/systemd/system/sn120-v2-bench.service
sudo systemctl daemon-reload
sudo systemctl enable --now sn120-v2-bench.service
journalctl -u sn120-v2-bench.service -f
```

For one run by hand in place of the unit: `bench once`. It benchmarks every waiting row, gives the node back and ends with `"event": "drained"`.

## 6. What you see when it is right

The log has one JSON line for each event. For the launch model, in this order:

| `event` | Meaning |
| --- | --- |
| `start` | The process runs. |
| `table-read` | It has read the published table. `published` is `true` when the store has one. |
| `rented` | A node is rented. The line has `pod`, `lium_id`, `gpu` and `usd_h`. `usd_h` is not above your cap. |
| `bootstrapped` | The software is on the node. `minutes` is the time since the rent. |
| `fetch` | The weights are on the node and each file's sha256 is checked. `status` is `rc0`. |
| `launched` | The run is started. `model` is `ref-qwen3.5-9b` first, then `genesis-87de8304e9d33dde`; `bids` names 6 benchmarks. |
| `table-written` | The table is in the store. `rows` shows the row as `evaluating`. |
| `collected` | The run is done. `scores` has a number for each of the 6 benchmarks. |
| `table-written` | `rows` shows the row as `scored`. |
| `released` | 900 seconds after the queue is empty, the node is given back. `hours` is the time that it was rented. |

Checks from any machine:

```
curl -s https://affine.io/public/benchmarks.json | python3 -c 'import json,sys; print([(m["model_id"], m["status"], m.get("benchmarks_done"), m.get("benchmarks_total")) for m in json.load(sys.stdin)["models"] if m.get("kind") in ("king", "genesis")])'
```

- It prints `[('genesis-87de8304e9d33dde', 'scored', 6, 6)]` after the first run.
- After each round that a miner wins, a row `king-<16 hex>` comes within one poll (30 seconds) of the publication of the new head. It goes from `queued` to `evaluating` to `scored`.
- The site's table shows the same rows. A state that is not `scored` is shown as a label on the row.
- `"$LIUM" ps` has no pod with the name `affine-v2-bench-1` once the log says `released`.

`bench status` shows the row in work, the queue, the fail counts and the rented pod at any time. It only reads, so it works while the unit runs.

## Public full rollouts and deploy order

The node also writes `OUT/<bench>/rollouts.jsonl.gz` from the verifiers message graph. Every expected sample has a row, including an explicit `error` row for a missing/failed rollout. The public file has only initial task chat messages, reference answers, sampled assistant text, grades and token/sampling metadata. It excludes runtime state, tool output, hostnames, IPs, pod ids and paths. Gzip is deterministic (`mtime=0`).

The host streams that file through ssh (50–150 MB must not be held in RAM), verifies the node-computed SHA-256, then uploads:
`public/rollouts/bench/<model_id>/<benchmark_id>.jsonl.gz` and its small `.meta.json` (`rows`, `bytes`, `sha256`, `bench_version`, UTC `finished_at`). A verified upload retires the disposable private download; R2 is the durable record. Transfer/store failures are logged as `rollouts-upload-deferred` and retried on later loops. They never prevent `summary.json`, `per_task.jsonl` collection or aggregate `public/benchmarks.json` publication. The bench store token needs write permission for these additional public keys.

**Deploy order: site pull first, then bench, then production between rounds.** Update `v2/site/pull.py` and let one pull succeed before starting the new bench service, so new object keys are accepted and the live index can be built. For the GPU-side head/winner capture in this change, stop production only at a closed round, update its checkout, and start the next round with the updated code. No live deployment is implied by this document.

See `v2/site/static/rollouts/SCHEMA.md` for field meanings. The hand-made backfill remains at its original paths; live paths do not replace it.

## 7. Stop, and the way back

```
sudo systemctl disable --now sn120-v2-bench.service
bench release
bench publish
"$LIUM" ps
```

- The stop alone does not give the node back. The run on the node goes on, and a later start collects it. This is right for a short stop.
- `release` gives the node back now. The log says `released`. A run that was in work is queued again at the next start, and it starts from the beginning on a new node.
- `publish` writes the row that was `evaluating` back to `queued`.
- The last command must list no pod with the name `affine-v2-bench-1`. If the unit was stopped in the minutes of a rent, a pod of that name can be there with no record: the service does not touch it, and it rents no other while it is listed. Remove it by hand in Lium.
- The table stays in the store and on the site with every result. Nothing else was changed: no round, no weight and no file of another service.
- To remove the unit: `sudo rm /etc/systemd/system/sn120-v2-bench.service && sudo systemctl daemon-reload`.

To move the service to another host: do the four lines above on the old host, then sections 4 and 5 on the new host. The new host reads the published table first, so no row runs again. The fail counts stay on the old host.

Run one service only. A second one on the same Lium account gets no node while the first has one. Each write starts from the table of the store, so two writers lose no result, but the state of a row can change back and forth.

## 8. Commands

| Command | What it does | While the unit runs |
| --- | --- | --- |
| `bench status` | Prints the row in work, the queue, the fail counts, the pod. Reads only. | Yes |
| `bench once` | Runs every waiting row, gives the node back, ends. | No |
| `bench publish` | Builds the table again and writes it to the store when it changed. Rents nothing. | No |
| `bench retry <model_id>...` | Puts a `failed` row back into the queue. | No |
| `bench release` | Gives the rented node back now. | No |

A command of the last four stops with `another affine-bench daemon holds ...` while the unit runs: stop the unit first.

## 9. When it is not right

| What the log says | Meaning | What to do |
| --- | --- | --- |
| `rent-no-offer` | No free node of the wanted kinds at or below the cap. No pod is made, and nothing is spent. The service asks again every 300 seconds. | Wait, or raise `BENCH_MAX_PRICE` with Jacob's word, then restart the unit. |
| `rent-refused`, `a pod named affine-v2-bench-1 already exists` | Lium lists a pod of that name and the service has no record of it. | Look at the pod in Lium. Remove it by hand if nobody uses it. |
| `rent-refused`, `8 rents in 24 h` | The limit of rents is reached. | Find out why nodes were lost. The limit is `BENCH_MAX_RENTS_PER_DAY`. |
| `bootstrap-failed` | The setup of the node failed. The node is given back. | Read `out` in the line. The service tries again at the next step. |
| `run-died`, `stalled` | The run ended with no result, or made no progress for 2400 seconds. The row's fail count goes up. After 3 fails the row is `failed` and is not tried again. | `bench retry <model_id>` after the cause is known. |
| `node-unreachable`, then `node-gone` | The node does not answer. After 10 minutes the service asks Lium. If the pod is gone, the row is queued again with no fail. | Nothing. |
| `release-deferred`, `release-failed`, `release-refused` | The node was not given back. It still costs money. | Check `"$LIUM" ps`. The service tries again at each step while it runs. |
| `error` with `InfraError` in `tb` | The store did not answer. The step is done again after 30 seconds. Before the first read of the table nothing is rented. | Check the token and the network if it stays. |

## 10. v1's benchmark service

v1's `affine-bench.service` is another service. It has another state directory, another pod name, and another file: `benchmarks.json` at the root of the site. v2's service reads and writes none of them.

On 14 Nov 2026 at 01:40 UTC, v1's file `https://affine.io/benchmarks.json` had a write time a few seconds old, and its `node` field named a pod `affine-bench-4` (8x B300, 32.0 dollars per hour) as up. So v1's service still ran then. When it stops is in the runbook (`v2/CUTOVER.md`), not here.

## 11. What is proven, and what is not

Proven by tests on a CPU (`v2/bench/tests/`), with the store of v2's simulation, a node that is a test double and a Lium command that is a test double:

- The rows come from public files only, and a public file that is missing makes no wrong row.
- The launch model runs once, each king runs once, and a row that is listed again is not run again.
- A host with an empty disk reads the published table and runs no row again. If the table cannot be read, nothing is rented.
- The only write to the store is `public/benchmarks.json`. The read links go to the node on standard input only, never into a command line or the log.
- The configured official reference runs first, followed by genesis and then kings newest first. One row runs at a time.
- Its pinned public Hugging Face download uses no HF write token, PAT or store credential.
- A reference row that is in the table before the first write stays, with its results. Once the official v2 Qwen row exists, only that Qwen row appears on home; the other stored row and its scores remain.

Not proven:

- No part of this service has run on a real GPU, on Lium or on R2 for v2. v1 ran the same node scripts on Lium.
- That vLLM 0.32.0 serves the launch model from the store's files with the settings of `bench/config/` is not shown by a run.
- The time and the price of one row on v2.
- That the Lium command and its key are on AffineValidator, and what the store token may do.
- That AffineValidator runs the site builder of this commit (section 4, step 6). On 14 Nov 2026 its `site-data.json` was not the output of the builder in `v2`.
- `systemd-analyze verify` passed for the unit file on the build machine. The unit has not run on a host.

Shown by a run of the commands on the build machine, with the store class of production (`R2Store`) on a local S3 double: `status`, `publish` and a start of `daemon` with a Lium double that has no offer. The start logged `rent-no-offer`, made no pod and wrote the table with the row `queued`.
