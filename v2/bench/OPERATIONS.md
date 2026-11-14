# The benchmark service of SN120 v2: operator steps

This file is for the operator. It says which host runs the service, which GPUs it uses, how to start and stop it, what you see when it is right, and the way back.

Words used here:

- **Row**: one model in the benchmark table. A row is the launch model (the genesis head), a king, or the last king of v1 (section 12).
- **King**: the model after a round that a miner won. The row is the folded head: the base with the winner's LoRA merged in.
- **Node**, **pod**: the GPU machine that the service rents on Lium for one row. Lium calls it a pod.
- **Worker**: the process of one row. The service starts it. It rents the row's node, runs the row and gives the node back.
- **Store**: the R2 bucket of v2.

## 1. What the service does

- It reads three kinds of public files from the store: `public/head.json`, `public/heads/<id>.json` and `public/rounds/<round>/round.json`. It calls nothing on the GPU box.
- From them it makes one row for the launch model and one row for each king. It adds one row for the last king of v1.
- A row with a missing result is queued. The rows run side by side: each row has a node of its own. At most 8 rows are in work at a time (`BENCH_MAX_PODS`); the newest queued row starts first when a place is free.
- A king's head can be public before its weight files are in the store. Before a row is queued, the service asks the store for each weight file that the head names (`public/bases/<base hash>/<file>`). If one is not there, the row waits: it gets no fail, no node is rented for it, and nothing is sent to a node. The service asks again at each poll, and the row is queued by the poll that finds every file. Nobody has to run a command.
- For a queued row the service starts a worker. The worker rents one node with the name `affine-v2-bench-<model_id>`, loads the row's weights from the store with read links that expire, runs the 10 benchmarks of `bench/config/benchmarks.toml`, and copies the scores.
- The worker gives the node back when the table in the store has the row's results. It asks the store for the table then; it does not trust the copy on the host.
- A worker waits only for its own rent, setup and download. No row holds a lock, a node or a file of another row. A row that cannot rent, or that fails, does not hold up the other rows.
- The service writes one object: `public/benchmarks.json`. It writes it on each change of a row or of a result. It deletes nothing in the store.
- The public pull on AffineValidator copies that object to the site, as it does today for the other public files. The site builder already reads it.
- The table is for information only. No code that picks a winner or sets weights reads it.

A row is run once. The service reads the published table before its first step. A result that is in the table is done, also on a host with an empty disk.

`<model_id>` is the id of the row: `genesis-`, `king-` or `v1king-` and the first 16 hex characters of the hash of the row's weights. Two rows never have the same id, also not at different times: another model has another hash.

**The first row.** Before 14 Nov 2026 the service had one node with the name `affine-v2-bench-1` and ran the rows one after the other. At the first start of this code, the row that runs on that node keeps it: the node's record (`BENCH_STATE/pod.json`), its ssh script (`BENCH_NODE_SSH`) and the run on it stay as they are. The service puts no new commit on that node and does not start its run again. When that row is done, the node is given back like any other.

**ArXivMath.** The settings ask for 2 samples for each of the 103 problems. The run of the first row has 16 samples for each problem. When it ends whole (16 samples for each of the 103 problems, none with an error), the worker computes the row's ArXivMath value from it by the fixed rule `pick-of-full/1` (`affine_bench/derive.py`): 2 of the 16 samples of each problem, picked by a hash of a constant and the problem's number. The result row then has `derived_from`, `derived_by`, `score_full`, `se_full` and `samples_full`. When the full run is not whole, the row gets no ArXivMath value: the log has one line `no-value` with the reason, the row is `held`, and `bench retry <model_id>` runs ArXivMath alone on a new node.

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
| **A. A node for each row, rented for the time of the row (the default)** | For each row: the price of its node for the setup, the download and the run. Never above `BENCH_MAX_PRICE` dollars per hour for one node. | No. It is another machine. It shares no GPU, no disk and no process with the round. |
| **B. Dublin's GPUs** | No rent. No round while a benchmark runs. | Yes. Not built, and not to be done by hand. |
| **C. A node that stays rented** | The price of the node for every hour, also with an empty queue. | No. Not built: the service has no setting for it. |

Details of A:

- **Which offer.** The service asks Lium for the offers of these kinds (`BENCH_GPU_PREF`, default `H100:8 H200:8 B200:8 B300:8`). An offer fits when it is a whole free node of 8 GPUs of one of these kinds, on the secure tier, at or below `BENCH_MAX_PRICE`. The worker takes the cheapest offer that fits. At one price it takes the kind that is earlier in the list. Reason: each of these kinds runs a row in about the same time, so the cheapest node gives the cheapest row. The kind and the price of the node are in the row (`models[].pod` in the table).
- **No cap, no rent.** Without `BENCH_MAX_PRICE` the service does not start.
- **One node for a row.** The node has the name `affine-v2-bench-<model_id>`. Before a rent the worker asks Lium for the pods of the account. If a pod has the row's exact name and the row has no record of it (an answer of Lium was lost, or the host lost its disk), the worker takes that pod as the row's node. It never rents a second one. If two pods have the row's name, the worker takes none and rents none: after 4 tries the row is `rent-failed` and the operator decides.
- **A rent that fails.** The worker tries again after 2, 4 and 8 minutes: 4 tries in all. Then the row has the state `rent-failed` in the table (`models[].stopped` is `rent_failed`), its worker ends and nothing more is spent for it. `bench retry <model_id>` queues it again. "No offer fits" is not a failed try: the worker asks again every 300 seconds, and no pod is made meanwhile.
- **Ceilings.** At most 8 rows are in work at a time. At most 16 pods are made in 24 hours, over all rows (`BENCH_MAX_RENTS_PER_DAY`). A pod lives 6 hours at most (`BENCH_POD_MAX_HOURS`): then the worker stops the run, copies the results that are finished, and gives the pod back; the row is `expired` until `bench retry`. The first row's node has no such limit. So the most that the service can spend in one hour is 8 times `BENCH_MAX_PRICE`.
- **A row that fails keeps no node.** After 3 fails, or when its pod is past 6 hours, the worker gives the node back. It waits 10 minutes at most for the store to have the results that it copied.
- **Deletes.** One place in the code sends a delete to Lium (`Pod.release` in `affine_bench/pod.py`). It sends it for the exact Lium id in the row's record, and only while Lium lists that id under the row's exact name. It never deletes from a name, a prefix or a list. It refuses the ids that start with `6cb913e4` and `c44cb565` and the names `sn120-v2-prod-dublin` and `casper-h100-ask-before-delete`, whatever a record says. It refuses the id that starts with `296c5237` (the first row's node) until the worker of that row has the row's results in the table of the store. A refused delete is in the log as `release-refused`; the node still costs money then.
- Each rent installs the software again and loads the weights again. The launch model's weights are 18.85 GB (the sum in `public/read_links.json`). Each node loads the weights of its row from the store by itself, so rows that start together load at the same time.
- The time and the price of one row are not in this file. v1's figures, from v1's public file `https://affine.io/benchmarks.json` on 14 Nov 2026: a median of 3582 seconds for one model (`progress.median_model_wall_s`), on a node of 8x B300 at 32.0 dollars per hour (`node`). v1 ran the same 10 benchmarks with the same harness.

Details of B:

- A round uses all 8 GPUs (`n_gpus = 8`). The benchmark's servers take 0.90 of each GPU's memory (`bench/config/serving.toml`).
- The round's evaluator starts only when no GPU holds more than 1024 MiB. It waits 120 seconds, then it stops with an error (`sn120_v2/gpu/evaluator.py`). So a round that meets a benchmark cannot score.
- Also without a clash, the time of a benchmark is time with no round. A round has at most 100 minutes (`round_time_limit_s = 6000`).
- The service has no setting for this choice. It always rents.

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

2. **The Lium command.** `"$LIUM" ps` lists the pods of the account and asks for nothing. The Lium key is in this user's Lium configuration, and Lium knows an ssh key of this user. No line of the list has a name that starts with `affine-v2-bench-`. (On a host that ran the service before 14 Nov 2026, the name `affine-v2-bench-1` can be there: it is the first row's node.)

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

   The output has `"queue_len": 1`, the id `genesis-87de8304e9d33dde` with `new-genesis` under `next`, and `"rows": {}` (no row is in work), while the head is the launch model and the table has no result. Under `waiting` it has `v1king-b6d592c3bb1b4393` with `the source is not set (BENCH_V1_KING_SOURCE)` (section 12). A store error here means that the token cannot read.

   Two more commands only read. `bench offers` prints the offers that a rent could take now: the kind, the price and the count, with no machine name. `bench pods` prints every bench pod of the Lium account with its row (section 8).

6. **The reference rows need no seed.** The service makes no row of a reference model, and none is put into its table. The site's builder (`v2/site/build.py`) takes the reference models from the frozen v1 history on AffineValidator, on each build, whatever this table holds. So start with an empty `$BDIR/site`: copy no file into it.

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

   It prints a line with `"event": "table-read"`, a line with `"event": "table-written"` whose `rows` has `"genesis-87de8304e9d33dde": "queued"`, `"v1king-b6d592c3bb1b4393": "waiting"`, one entry for each other id that step 5 listed and no reference model, and then `published`. A store error here means that the token cannot write.

   After the pull's next run (one minute): `curl -s -o /dev/null -w '%{http_code}\n' https://affine.io/public/benchmarks.json` prints `200`. The site's table has the same reference models as before, and its row `Genesis` has the state `queued` in place of `no run yet`.

   If the pull's log says `public/benchmarks.json is in the site directory and this command did not write it`, a file placed by hand is in its way (step 6): move it out of the site directory. The next run of the pull copies the table of the store.

## 5. Start

The start rents a node for each queued row, as soon as the row is queued. After section 4 that is at once.

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

For one run by hand in place of the unit: `bench once`. It benchmarks every queued row, gives every node back and ends with `"event": "drained"`.

## 6. What you see when it is right

The log has one JSON line for each event. A line of a worker has `row`, the id of its row. For the launch model, in this order:

| `event` | Meaning |
| --- | --- |
| `start` | The service runs. |
| `table-read` | It has read the published table. `published` is `true` when the store has one. |
| `row-worker`, `row-start` | The service has started the worker of a row. |
| `rented` | The row's node is rented. The line has `pod`, `lium_id`, `gpu` and `usd_h`. `usd_h` is not above your cap. |
| `bootstrapped` | The software is on the node. `minutes` is the time since the rent. |
| `fetch` | A download to the node has ended. `out` has `"status": "materialized"` (or `"present"`) when the weights are on the node and each file's sha256 is right. `status` is `rc0` in both cases, also when the download failed: read `out`. |
| `launched` | The run is started. `row` is `genesis-87de8304e9d33dde`, and `bids` names 10 benchmarks. |
| `table-written` | The table is in the store. `rows` shows the row as `evaluating`. |
| `collected` | The run is done. `scores` has a number for each of the 10 benchmarks. |
| `table-written` | `rows` shows the row as `scored`. |
| `released` | The table in the store has the row's results, and the node is given back. `hours` is the time that it was rented. |
| `row-end` | The worker has ended. |

Checks from any machine:

```
curl -s https://affine.io/public/benchmarks.json | python3 -c 'import json,sys; print([(m["model_id"], m["status"], m.get("benchmarks_done"), m.get("benchmarks_total")) for m in json.load(sys.stdin)["models"] if m.get("kind") in ("king", "genesis")])'
```

- It prints `[('genesis-87de8304e9d33dde', 'scored', 10, 10)]` after the first run.
- After each round that a miner wins, a row `king-<16 hex>` comes within one poll (30 seconds) of the publication of the new head. It goes from `queued` to `evaluating` to `scored`.
- A row whose weight files are not in the store yet has `"waiting_for": "weights"` in the table. Its `status` stays `queued`. When the files are still not there more than 60 minutes after its head was applied, its `status` is `waiting`. The log has one line `waiting` when that starts, one line `waiting-late` at the 60 minutes, and one line `waiting-over` when the files are there. `bench status` lists such rows under `waiting`, with the name of a file that is not there.
- While a row is `evaluating`, its `progress` in `models[]` says about each 5 minutes what its worker saw on its node (the top-level `progress` has the same fields for the newest row in work): `checked_at` (the time of the last step in which the node answered), `output_lines` (finished rollouts), `output_grew_at` (when that count last grew) and `benchmarks_finished` (benchmarks of the row that have ended on the node; `benchmarks_done` on the row counts collected results and stays 0 until the whole row is collected). So `updated_at` moves about each 5 minutes while a row runs. A `checked_at` that is older than 10 minutes while a row is `evaluating` means that its node does not answer, that its worker does not step, or that the service cannot write the table.
- The site's table shows the same rows. A state that is not `scored` is shown as a label on the row.
- `models[].pod` of a row says its node: `name` (Lium's name of the machine), `gpu`, `usd_h`, `state` (`renting`, `setup`, `running`, `released`, `rent-failed`) and `since`. The top-level `node` object is the record `BENCH_STATE/pod.json`, as before: the first row's node, and `up` is `false` once that node is given back.
- `bench pods` lists no pod of a row once the log says `released` for that row.

`bench status` shows the rows in work with their state, the queue, the rows that wait, and the fail counts at any time. It only reads, so it works while the unit runs.

## 7. Stop, and the way back

```
sudo systemctl disable --now sn120-v2-bench.service
bench pods
bench release <model_id>...
bench publish
bench pods
```

- The stop ends the service and the workers. It alone gives no node back. The run on each node goes on, and a later start collects it. This is right for a short stop.
- A stop in the minutes of a rent can leave a pod that the row has no record of. Nothing is lost: the pod has the row's exact name, so the worker of the next start takes it and rents no other.
- `bench pods` lists every bench pod with its row. `release` gives the node of each named row back now, by the exact id of the row's record. The log says `released`. A run that was in work is queued again at the next start, and it starts from the beginning on a new node.
- `release` does not give the first row's node (`296c5237`) back while that row is not done: the log says `release-refused`. To end that node before its row is done is the operator's own act in Lium.
- `publish` writes the rows that were `evaluating` back to `queued`.
- The last command must list no pod. A pod that it marks `NO ROW CLAIMS THIS POD` is not touched by the service: look at it in Lium and decide.
- The table stays in the store and on the site with every result. Nothing else was changed: no round, no weight and no file of another service.
- To remove the unit: `sudo rm /etc/systemd/system/sn120-v2-bench.service && sudo systemctl daemon-reload`.

To move the service to another host: do the lines above on the old host, then sections 4 and 5 on the new host. The new host reads the published table first, so no row runs again. The fail counts stay on the old host.

Run one service only. A second one on the same Lium account finds the pods of the first under the rows' names and would take them. Each write starts from the table of the store, so two writers lose no result, but the state of a row can change back and forth.

## 8. Commands

| Command | What it does | While the unit runs |
| --- | --- | --- |
| `bench status` | Prints the rows in work with their state, the queue, the rows that wait, the fail counts. Reads only. | Yes |
| `bench offers` | Prints the offers that a rent could take now: kind, price, count. Reads only. | Yes |
| `bench pods` | Prints every pod of the Lium account whose name starts with `affine-v2-bench-`, and the pod of each row's record: the first 8 characters of the id, the name, the row, the row's state, the age in hours, the price. It marks a pod that no row claims, a record whose pod Lium does not list, and a row in work that has no pod. It prints no address. Reads only: it deletes nothing. | Yes |
| `bench retry <model_id>...` | Puts a row that is `failed`, `held`, `rent-failed` or `expired` back into the queue. | Yes, when the row's worker has ended |
| `bench once` | Runs every queued row, gives every node back, ends. A row that waits for weight files is left for a later run. | No |
| `bench publish` | Builds the table again and writes it to the store when it changed. Rents nothing. | No |
| `bench release <model_id>...` | Gives the node of each named row back now. | No |

A command of the last three stops with `another affine-bench daemon holds ...` while the unit runs: stop the unit first.

## 9. When it is not right

| What the log says | Meaning | What to do |
| --- | --- | --- |
| `rent-no-offer` | No offer fits now (section 3). No pod is made, and nothing is spent. The worker asks again every 300 seconds. | Wait, or raise `BENCH_MAX_PRICE` with Jacob's word, then restart the unit. |
| `rent-failed` | A try of a rent failed: Lium refused it (`why` has Lium's words, for example a name that it does not take), the setup failed, or more than one pod has the row's name. After 4 tries the line has `"gives_up": true`, the row is `rent-failed` and its worker ends. The other rows go on. | Read `why`. `bench pods`. `bench retry <model_id>` after the cause is known. |
| `pod-taken-by-name` | Lium lists a pod under the row's exact name that the row had no record of. The worker took it as the row's node. | Nothing. |
| `rent-refused`, `16 pods were made in 24 h` | The limit of rents is reached. | Find out why nodes were lost. The limit is `BENCH_MAX_RENTS_PER_DAY`. |
| `bootstrap-failed` | The setup of the node failed. The node is given back. It counts as a failed try of the rent. | Read `out` in the line. |
| `run-died`, `stalled`, `run-incomplete` | The run ended with no result, made no progress for 2400 seconds, or ended without the result of a benchmark. The row's fail count goes up. After 3 fails the row is `failed`, its node is given back and it is not tried again. | `bench retry <model_id>` after the cause is known. |
| `pod-expired` | The row's pod is older than 6 hours. The run is stopped, the finished results are copied, the pod is given back. The row is `expired`. | Find out why the row was slow. `bench retry <model_id>` runs what is missing on a new node. |
| `no-value` | A run with more samples than the settings ask for was not whole, so no value at the settings' count was computed (section 1). The row is `held`. | Read `why`. `bench retry <model_id>` runs that benchmark alone on a new node. |
| `node-unreachable`, then `node-gone` | The node does not answer. After 10 minutes the worker asks Lium. If the pod is gone, the row is run again on a new node with no fail. | Nothing. |
| `release-waits` | The row is done and the table in the store does not have its results yet. The node stays until it has. A row that failed waits 10 minutes at most. | If it stays: the service cannot write the table. Check the token. |
| `release-deferred`, `release-failed`, `release-refused` | The node was not given back. It still costs money. | `bench pods`. The worker tries again every 10 minutes while the service runs. `release-refused` names the guard (section 3). |
| `waiting`, then `waiting-late` | A row's weight files are not in the store. The row waits with no fail and no node. `waiting-late`: they are still not there more than 60 minutes after its head was applied, and the table shows the row as `waiting`. | Nothing for `waiting`. For `waiting-late`: find out why the GPU box has not sent the files (`file` in the line names one). The row starts by itself when they are there. `bench retry` does not help. |
| `fetch` whose `out` has no `materialized`, and the row's fail count goes up | The files are in the store and the download to the node failed: a sha256 that is not the head's, a link that the store refuses, a full disk. After 3 fails the row is `failed`. | Read `out`. `bench retry <model_id>` after the cause is known. |
| `store-check-failed` | The store did not answer the question whether a row's weight files are there (a token that may not ask, the network). The row is not held back: the download on the node decides, and a failure there counts as a fail. | Check the token if it stays. |
| `error` with `InfraError` in `tb` | The store did not answer. The step is done again after 30 seconds. Before the first read of the table nothing is rented. | Check the token and the network if it stays. |

## 10. v1's benchmark service

v1's `affine-bench.service` is another service. It has another state directory, another pod name, and another file: `benchmarks.json` at the root of the site. v2's service reads and writes none of them.

On 14 Nov 2026 at 01:40 UTC, v1's file `https://affine.io/benchmarks.json` had a write time a few seconds old, and its `node` field named a pod `affine-bench-4` (8x B300, 32.0 dollars per hour) as up. So v1's service still ran then. When it stops is in the runbook (`v2/CUTOVER.md`), not here.

## 11. What is proven, and what is not

Proven by tests on a CPU (`v2/bench/tests/`), with the store of v2's simulation, a node that is a test double and a Lium command that is a test double:

- The rows come from public files only, and a public file that is missing makes no wrong row.
- The launch model runs once, each king runs once, and a row that is listed again is not run again.
- Each row has a pod of its own with the name `affine-v2-bench-<model_id>`, and the rows run side by side. A row that cannot rent holds up no other row.
- A pod is given back only when the table in the store has the row's results, and by the exact id of the row's record.
- A rent takes the cheapest whole secure node of an allowed kind at or below the cap. A lost answer of Lium makes no second pod. A rent that fails is tried 4 times with a growing wait; then the row says so and the other rows go on.
- No delete is sent for a protected id or name, for a pod of another name, or for the first row's node before that row is done.
- The ceilings: the count of rows in work, the count of pods in 24 hours, the longest life of a pod. A row that fails 3 times, or hangs, keeps no pod.
- The first row keeps its node; nothing is put on it; its run is not started again; its ArXivMath value is the pick of 2 of 16, only from a whole run.
- The row of the last king of v1: it is queued once, it rents nothing without a readable source, it changes no king row and not the kings' order, and a download that fails gives its pod back by its exact id.
- A host with an empty disk reads the published table and runs no row again. If the table cannot be read, nothing is rented.
- The only write to the store is `public/benchmarks.json`. The read links go to the node on standard input only, never into a command line or the log.
- A reference row that is in the table before the first write stays, with its results.
- A king whose weight files are not in the store gets no fail, no rent and no command on the node, and starts by itself when they are there. A download that fails on the node while the files are in the store still counts as a fail. A restart of the service in a run starts no second run and rents no second node.
- The site builder (`v2/site/build.py`) reads the table as it is, and adds the reference models from the frozen v1 history (`v2/tests/test_site_build.py`, `v2/tests/test_site_pull.py`).

Not proven:

- The code of this commit has not run on Lium. The single-node code of before ran rows on Lium and R2 for v2.
- That Lium takes a pod name of 37 to 40 characters (`affine-v2-bench-<model_id>`). The longest name that is known to work has 29. Lium's command checks no name itself; its server decides. If Lium refuses the name, no pod is made and the row is `rent-failed` with Lium's words.
- That four rows can load their weights from the store at the same time at full speed.
- That vLLM 0.32.0 serves the launch model from the store's files with the settings of `bench/config/` is not shown by a run.
- The time and the price of one row on v2.
- That the Lium command and its key are on AffineValidator, and what the store token may do.
- That AffineValidator runs the site builder of this commit (section 4, step 6). On 14 Nov 2026 its `site-data.json` was not the output of the builder in `v2`.
- `systemd-analyze verify` passed for the unit file on the build machine. The unit has not run on a host.

Shown by a run of the commands on the build machine, with the store class of production (`R2Store`) on a local S3 double: `status`, `publish` and a start of `daemon` with a Lium double that has no offer. The start logged `rent-no-offer`, made no pod and wrote the table with the row `queued`.

## 12. The last king of v1

The table has one row for the last king of v1: `v1king-b6d592c3bb1b4393`, kind `v1-king`, UID 920, v1 epoch 1211, v1 checkpoint `b6d592c3bb1b439365e82147a47e65d93c083a13444aceae078f8a2053cd5ce3` (v1's public record: `https://affine.io/v1-data.json`). The service runs it like a king row: the same 10 benchmarks, the same settings and budget (v1's benchmark service served its kings with the same profile, `qwen3.5-family`), a node of its own, the same guards and ceilings. It has no time of a v2 head, so it is the last row of the queue, and it changes no king row.

Its weights are not in v2's public files. The operator says where they are, with one setting in the settings file:

```
BENCH_V1_KING_SOURCE=<a prefix in the store of the service>
```

- Under that prefix the store holds `descriptor.json` and each weight file under its plain name. `descriptor.json` is `{"id": "<the v1 checkpoint>", "files": {"<file name>": "<sha256 of the file>", ...}}`. This is the form of v1's publication record. `id` must be the checkpoint above.
- This is the one form that the service takes today. It does not read a Hugging Face repository. The setting holds no secret: the service reads the files with its store token and gives the node read links that expire.
- While the setting is empty, or `descriptor.json` or a file cannot be read, the row has the status `waiting` with `"waiting_for": "source"` in the table, and `bench status` says why under `waiting`. Nothing is rented for it, and it holds up no other row.
- The service reads the setting at its start. After a change of the settings file, restart the unit. The row is queued by the first step that can read the source.
- The node checks the sha256 of each file after the download. A download that fails counts as a fail; after 3 fails the row is `failed` and its node is given back.

Not proven: that vLLM serves the v1 checkpoint's files from the store on a v2 bench node. v1's service served the same checkpoint with the same serving settings.
