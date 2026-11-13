# SN120 v2 smoke test: one real round on one secure 8x B300

Jacob runs this. Do the steps in order. Each step shows the command, the output to expect, and what counts as a pass.

Nothing here sets chain weights. Nothing here writes to a production key. Nothing here is the cutover: that is [CUTOVER.md](CUTOVER.md), after this test passes.

## Words used here

- **Box**: the 8x B300 machine.
- **Store**: the R2 bucket. Every v2 key is under one prefix, `SN120_R2_PREFIX`.
- **Journal**: the controller's only memory, a list of records. Each record goes to the store first (`private/journal/`), then to the local file `journal.jsonl`.
- **State directory**: the local folder with `journal.jsonl`. It is a cache. A new box with an empty folder continues from the store.
- **Stack**: the real parts (store, admission, sampler, trainer, evaluator, folder, publisher). `--stack sn120_v2.assembly:stack` builds them from environment variables.
- **Head (M)**: the current model. **Null M**: the head with no LoRA, scored as the baseline.
- **Round**: take up to 32 queued batches, train one LoRA per batch, score every LoRA and null M on the same tasks, apply at most one winner (z > 1), fold it into the head.
- **z**: the gain of a LoRA over null M, divided by its standard error.
- **Update norm**: the size (Frobenius norm) of a trained LoRA's change to the model weights. `recipe.global_norm_cap` is a limit on it.

## Safety rules of this test

- The box holds no wallet. It only reads the chain. The wallet is on AffineValidator, a separate machine. This test does not use that machine.
- `SN120_R2_PREFIX` must be set, and must be a prefix that production does not use. Use a separate bucket if you have one.
- The v1 bucket is read with a read-only token, in step 5 only.
- The weights step is `--dry-run` only.

## 0. Before the box: CPU check (any machine, no GPU, no network, about 1 minute)

```
git fetch origin && git checkout origin/v2
git rev-parse HEAD            # write this down
cd v2
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,r2,chain,envs,miner]"
python -m pytest
python -m sn120_v2 sim --state-dir /tmp/sn120-sim --rounds 6 --miners 4
```

Pass: `pytest` shows no failure. `sim` prints `"replay_matches": true` and exits with code 0.

## 1. Setup of the box

The box needs: Linux with the NVIDIA driver (`nvidia-smi` works), 8 GPUs, Python 3.12, `uv`, 300 GB of free local disk, 64 GB of RAM, and free ports 8100 to 8107 on `127.0.0.1`. There is no baked image: the box is a plain CUDA machine plus the install line.

```
git clone https://github.com/AffineFoundation/affine.git && cd affine
git checkout <the commit of step 0>
cd v2
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,gpu,r2,chain,envs,miner]"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
python -m pytest
```

Pass: `nvidia-smi` lists 8 GPUs with almost no memory in use. `pytest` shows no failure.

Set the paths. Every later step uses them. Run every later command from `v2/`, inside the virtual environment.

```
export SMOKE=/data/sn120-smoke            # a local disk with 300 GB free
export STATE=$SMOKE/state                 # the state directory
export SN120_WORK_DIR=$SMOKE/work         # model cache, job files, logs. Disposable.
mkdir -p $SMOKE $SN120_WORK_DIR
```

The two digests. The **image digest** is a hash of the Python version and of every installed package. The **code digest** is a hash of the GPU code. Both go into the parameter file, so a box with other packages or other code cannot train or score.

```
python -m sn120_v2.gpu.plan | tee $SMOKE/plan.json
export SN120_IMAGE_DIGEST=$(python -c "import json,os; print(json.load(open(os.environ['SMOKE']+'/plan.json'))['image_digest'])")
python - <<'EOF'
import json, os
plan = json.load(open(os.environ["SMOKE"] + "/plan.json"))
params = {"batch_cap": 8, "recipe": {"image_digests": [plan["image_digest"]], "code_digest": plan["code_digest"]}}
json.dump(params, open(os.environ["SMOKE"] + "/params.json", "w"), indent=1)
print(json.dumps(params, indent=1))
EOF
```

Pass: `plan.json` is one line, `{"code_digest": "<64 hex>", "image_digest": "<64 hex>"}`, and `params.json` holds the same two values.

`"batch_cap": 8` makes the first round small: at most 8 batches, one per GPU, all trained at the same time. Everything else is the launch value: 2000 tasks, 40 minutes of training time, at most 100 minutes for the round. For a round of launch size, leave `"batch_cap": 8` out (the launch value is 32). The file sets no `global_norm_cap`: the smoke test runs with no cap and measures the norm.

For a first round that takes minutes, not hours, see "Tiny first" after step 4.

`bootstrap` (step 4) pins these parameters. A change of `params.json` after step 4 has no effect by itself. To change a parameter later: stop `run`, edit the file, then run `python -m sn120_v2 set-params --state-dir $STATE --stack sn120_v2.assembly:stack --params $SMOKE/params.json`. It prints the old and the new `params_hash`.

After any `uv pip install` or `git pull`, run this block again. A stale digest stops the first train call with an error; it never gives a wrong score.

The model download is part of step 3.

## 2. R2: the smoke prefix and the self-test

Take the four secrets from your secret store. Do not write them to a file in the repository.

```
export SN120_R2_ENDPOINT=...              # https://<account id>.r2.cloudflarestorage.com
export SN120_R2_BUCKET=...
export SN120_R2_ACCESS_KEY_ID=...
export SN120_R2_SECRET_ACCESS_KEY=...
export SN120_R2_PREFIX=smoke-$(date -u +%Y%m%dT%H%M)
echo $SN120_R2_PREFIX                     # write this down; use the same value in every shell
: "${SN120_R2_PREFIX:?the smoke prefix is not set}"
export SN120_PUBLIC_URL=none              # the smoke test has no public address
export SN120_V2_CHAIN_NETWORK=finney      # read-only chain reads; the box has no wallet
python -m sn120_v2.r2 --self-test
```

Expected: 9 lines that start with `ok`, then `PASS  r2 self-test, keys under smoke-.../selftest/...`. The test writes small keys and one 11 MiB object under the prefix, checks them, and deletes them.

Pass: the last line starts with `PASS`.

## 3. Genesis: build and upload the base and the master

The **base** is the model as Hugging Face has it (18.8 GB). The **master** is the same weights in full precision, fp32 (37.6 GB). Each fold adds the winner's LoRA to the master.

```
python -m sn120_v2.gpu.genesis --out $SMOKE/genesis --upload 2>&1 | tee $SMOKE/genesis.log
export MASTER_DIGEST=$(tail -n 1 $SMOKE/genesis.log | python -c "import json,sys; print(json.load(sys.stdin)['master_digest'])")
echo $MASTER_DIGEST
```

The command downloads `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` at the pinned revision (no token needed), checks it against the base hash in the code, builds the master, uploads both under the smoke prefix, and reads each uploaded file back to compare its hash. Time: not measured; expect 20 to 40 minutes. No GPU is used. A second run does not download or upload again.

Expected last line:

```
{"base_hash": "87de8304...", "base_dir": "...", "master_digest": "<64 hex>", "master_dir": "...", "genesis_files": ".../genesis/genesis_files.json", "uploaded_files": <count>}
```

Pass: that line is printed, `uploaded_files` is above 0, and `echo $MASTER_DIGEST` prints 64 hex characters.

Optional, and it saves time: put the two file sets into the model cache of the box now. Without this block, the first round downloads the 56 GB again from R2 before it trains.

```
export BASE_HASH=$(tail -n 1 $SMOKE/genesis.log | python -c "import json,sys; print(json.load(sys.stdin)['base_hash'])")
mkdir -p $SN120_WORK_DIR/cache
cp -alT $SMOKE/genesis/base $SN120_WORK_DIR/cache/$BASE_HASH
cp -alT $SMOKE/genesis/master $SN120_WORK_DIR/cache/$MASTER_DIGEST
```

`cp -alT` makes hard links: a second name for the same file, with no second copy on the disk. It needs `$SMOKE/genesis` and `$SN120_WORK_DIR` on the same disk. If it prints `Invalid cross-device link`, use `cp -aT` (a real copy, 56 GB). The box hashes every file in the cache before it uses it. It deletes a directory that does not match and downloads it from R2; the files in `$SMOKE/genesis` stay as they are.

Pass: `ls $SN120_WORK_DIR/cache` shows two directories, with the names `$BASE_HASH` and `$MASTER_DIGEST`.

## 4. Import dry run from v1, then bootstrap

The import reads two public v1 files (`https://affine.io/k1.json` and `https://affine.io/mining.json`) and one chain snapshot. It needs no credential and writes nothing to v1 or to the chain.

```
export HK=5HmYnmUYT6qe3yFMg1Ad8WLLqDvnwtjYakXBpDvoRW1Qqzb8     # the validator hotkey; also the hotkey of UID 0
python -m sn120_v2.chain --netuid 120 --network finney --out $SMOKE/chain_snapshot.json
python -m sn120_v2.cutover --dry-run --chain $SMOKE/chain_snapshot.json --owner-hotkey $HK \
    --earlier-k1 tests/data/cutover/k1.json
```

Expected from the dry run: the 10 kings, the front seats (the v1 turns with no verdict), the start queue, a `first weights` line with 10 UIDs, and a last line `UID 0 is 5HmY... it is the given owner hotkey`. It writes nothing.

Pass: no `CUTOVER IMPORT REFUSED`. The 10 kings are the miners that v1 pays now.

Now write the import files and start v2's state:

```
python -m sn120_v2.cutover --out $SMOKE/cutover --chain $SMOKE/chain_snapshot.json \
    --master-digest $MASTER_DIGEST --validator-hotkey $HK --owner-hotkey $HK \
    --earlier-k1 tests/data/cutover/k1.json
python -m sn120_v2 bootstrap --state-dir $STATE --params $SMOKE/params.json --stack sn120_v2.assembly:stack \
    --genesis $SMOKE/cutover/genesis.json --genesis-files $SMOKE/genesis/genesis_files.json \
    --v1-import $SMOKE/cutover/v1_import.json --chain $SMOKE/cutover/chain.json
```

`bootstrap` does this, in order: it checks every genesis file in the store against its sha256 (56 GB are read back, so this takes minutes); it writes the genesis, the pinned parameters, the 10 kings and the queue to the journal; it publishes `llms.txt` and `mining.json`.

Expected output, a JSON summary:

```
{"phase": "IDLE", "head": "<64 hex>", "head_folded": true, "heads": 1, "rounds_closed": 0, "last_outcomes": [],
 "queue": <seats>, "served": <count>, "kings": [<10 UIDs>], "params_hash": "...", "journal_records": <count>, "published": true}
```

Pass: exit code 0; `kings` are the 10 UIDs of the dry run; `queue` is the number of seats the dry run printed; `published` is `true`.

The parameters are now **pinned**: they are a record in the journal, and every process uses that record, not a local file. A second `bootstrap` on the same prefix is refused.

## Tiny first: five rounds, small to full

Do this after step 4. It uses the one bootstrap of step 4 for five rounds. Each round is a **stage** with its own parameter file. `set-params` pins that file before the round. The first stage runs the whole path in minutes. The last stage is the full round of steps 5 to 9.

| Stage | Batches | GPUs in use (`n_gpus`) | Tasks | Reply cap (`max_output_tokens`) | What it adds | Expected time |
|---|---|---|---|---|---|---|
| `1-tiny` | 1 | 1 | 1 | 2048 | one training job on one GPU, one vLLM server, scoring, decision, publish | 10 to 20 minutes the first time, 5 to 8 after that |
| `2-two` | 2 | 2 | 8 | 2048 | two training jobs at the same time, three looks | 6 to 10 minutes |
| `3-eight` | 8 | 8 | 32 | 65,536 (the launch value) | 8 jobs, 8 servers, a winner, promotion, fold | 25 to 45 minutes |
| `4-multi` | 2 | 8 | 8 | 2048 | 4 GPUs for each training job, on a folded head | 8 to 15 minutes |
| `5-full` | 32 | 8 | 2000 | 65,536 | the launch values | up to 100 minutes, then the fold |

Where the minutes come from (estimates, not measured): each `run` hashes the model files (1 minute) and starts the vLLM servers (2 to 4 minutes, 4 to 8 the first time). The first round downloads the model into the box cache (3 to 10 minutes). One training job is 1 to 1.5 minutes, plus a one-time compile on each GPU. A block of tasks lasts as long as its longest reply: about 1 minute at 2048 tokens, 11 to 18 minutes when one reply runs to 65,536. A fold is 8 to 16 minutes.

The four small files also set these values. They are for the smoke test only:

- `z_bar` 0 and `min_scored_tasks` 2: a LoRA wins when it is above null M on at least 2 tasks (z above 0). So a small round can have a winner.
- `max_infra_attempts` 1000: no seat is put on hold while faults are fixed.
- `apply_retry_s` 5: after a failed round, the next one starts 5 seconds later, not 5 minutes.

`5-full` holds only the two digests: every other value is the launch value. The mainnet bootstrap has its own file ([CUTOVER.md](CUTOVER.md)); never give it one of the small files.

Write the files. Run this again each time the digest block of step 1 ran:

```
python -m sn120_v2.smoke params --plan $SMOKE/plan.json --out $SMOKE
export V1_R2_ENDPOINT=... V1_R2_BUCKET=... V1_R2_ACCESS_KEY_ID=... V1_R2_SECRET_ACCESS_KEY=...   # read-only token
```

Expected: seven lines. Five are the stages, `$SMOKE/params-1-tiny.json` to `$SMOKE/params-5-full.json`. Two are for the rounds that hit the time limit on purpose (below): `params-limit-judged.json` and `params-limit-not-scored.json`.

Each stage is the same three commands. Run them for `1-tiny`, then `2-two`, `3-eight`, `4-multi`:

```
export STAGE=1-tiny
python -m sn120_v2 set-params --state-dir $STATE --stack sn120_v2.assembly:stack --params $SMOKE/params-$STAGE.json
python -m sn120_v2.smoke copy --state-dir $STATE | tee -a $SMOKE/copied.txt
python -m sn120_v2 run --state-dir $STATE --stack sn120_v2.assembly:stack --rounds 1 2>&1 | tee $SMOKE/run-$STAGE.log
```

For `5-full`, the copy command is `python -m sn120_v2.smoke copy --state-dir $STATE --whole`. Then do the `status` check of step 5 and steps 6 to 9 as written.

What the three commands do:

1. `set-params` prints the old and the new `params_hash`. It is refused while a round is open. `n_gpus` changes only here, between two `run` commands.
2. `copy` gives the next seats of the queue a miner file: as many as the round takes (`batch_cap`). It reads v1's bucket and writes only under the smoke prefix. Without `--whole` it writes a **small copy**: at most 2 groups of the real file. A group is one prompt with its rollouts. Each group is copied whole and unchanged. A group is used only when it has two different rewards and every rollout of it is within the reply cap. The copy has no `train` object, so it trains with the default recipe (1 step). The copy must pass the full admission check, or it is not written.
3. `run` runs one round and prints the round summary.

Expected from `copy`: one line for each file, for example `copied 5F.. 2 of 61 groups, 16 rollouts, longest 1890 tokens, 51234 of 48211002 bytes sha256=... default recipe`. Then `1 of 1 files copied`. Lines that start with `passed over` name a file with no usable group, and why.

If `copy` ends with fewer files than it needs (exit code 1): the real files have too few short groups for this cap. The `passed over` lines say `the nearest group needs <N>`. Write the files again with a cap above that N, then run the three commands again:

```
python -m sn120_v2.smoke params --plan $SMOKE/plan.json --out $SMOKE --short-reply 4096
```

Use the same `--short-reply` each time you write the files after that.

Pass, for each stage:

- `1-tiny`: the summary starts with `SN120 v2 round r00000N: NULL`. One row, verdict `lost`, `tasks scored: 1 of 1`. `NULL` is the only correct result: one task has no standard error, so this round can have no winner. The JSON summary shows `"phase": "IDLE"`.
- `2-two`: `NULL` or `APPLIED`. Two rows, each `lost` or `winner`. With the cap of 2048 most replies are cut, so expect `NULL`.
- `3-eight`: `APPLIED`, 8 rows, a `winner` line, `"heads"` is 2 more than before, `"head_folded": true`, and `kings` holds the winner's UID. The chance of a winner is about 80 to 93 percent, when null M passes between 5 and 90 percent of the 32 tasks: with `z_bar` 0, one of 8 LoRAs is above null M by chance alone. If the result is `NULL`, that is not a fault: run the three commands of this stage again. They take the next 8 seats.
  One case to know: every row shows `delta +0.0000`. Then the small copies trained an update too small to change one reply. Run the stage again with whole files: use `python -m sn120_v2.smoke copy --state-dir $STATE --whole` as the copy command. Whole files train a real update. Some of them can be invalid under the v2 rules, so the round can then have fewer than 8 rows.
- `4-multi`: two rows, each `lost` or `winner`. The two `"phase": "train"` lines of this round in `$SN120_WORK_DIR/logs/phases.jsonl` are the first trainings with 4 GPUs each.
- `5-full`: the pass lines of steps 6 to 9.

In every stage: no row has the verdict `no_decision`, `not_trained` or `not_scored`, and the first line is not `NO_DECISION`.

### A round that hits the time limit on purpose

This stage is optional, and its place is between `4-multi` and `5-full`. It proves the rule at the round's time limit on the real box. It is two rounds. Each one is `2-two` (2 batches, 2 GPUs, reply cap 2048) with a test of 2000 tasks and a short time limit. 2000 tasks can never be scored inside these limits, so each round hits its limit.

The rule (step 6 has it in full): the box stops scoring at the **stop time**. That is the time limit minus `promote_reserve_s`, counted from the lease. A block of tasks that still runs then is stopped and is not used. Then:

- A LoRA with the first look complete is judged on the blocks that null M and every open LoRA finished. It is charged, win or lose.
- A LoRA with fewer scored tasks than the first look is `not_scored`. It is not charged and it is first in the next round.

| Stage | Proves | Time limit (`round_time_limit_s`) | Stop time | Training time (`train_time_budget_s`) | First look | Block (`eval_chunk_tasks`) | Expected time |
|---|---|---|---|---|---|---|---|
| `limit-judged` | the judgment on the blocks finished | 1500 s (25 minutes) | minute 23 | 600 s | 4 tasks (`looks` 0.002 and 1) | 4 tasks | 26 to 28 minutes |
| `limit-not-scored` | `not_scored`: not charged, place kept | 480 s (8 minutes) | minute 6 | 300 s | 2000 tasks (`looks` 1) | 2000 tasks | 9 to 11 minutes |

Other values of the two files: `promote_reserve_s` 120 in both, so the box stops 2 minutes before the time limit. `lease_timeout_s` is 2 times the time limit (3000 and 960): a result that arrives later than this is not used. `limit-judged` also sets `z_bar` 1 (the launch win rule) and `min_scored_tasks` 8: with fewer than 8 scored tasks no LoRA can win, and no LoRA is dropped at the first look of 4 tasks.

Run `limit-judged` first. It is the main proof: it also shows that the box stops a running block at the stop time. Run it only after `1-tiny` and `2-two` passed on this box: the first replies of a new vLLM server can take minutes, and `limit-judged` leaves the first block that time. Then run `limit-not-scored`. Each round is the same three commands as a stage:

```
export STAGE=limit-judged
python -m sn120_v2 set-params --state-dir $STATE --stack sn120_v2.assembly:stack --params $SMOKE/params-$STAGE.json
python -m sn120_v2.smoke copy --state-dir $STATE | tee -a $SMOKE/copied.txt
python -m sn120_v2 run --state-dir $STATE --stack sn120_v2.assembly:stack --rounds 1 2>&1 | tee $SMOKE/run-$STAGE.log
```

Pass for `limit-judged`, all of these:

1. The first line ends with `(stopped at the time limit)`: `SN120 v2 round r00000N: NULL (stopped at the time limit)` or `... APPLIED (stopped at the time limit)`. Both are a pass.
2. The line `batches: 2 | tasks scored: N of 2000`. N is a multiple of 4 and far below 2000. Expect some tens.
3. Two rows, each `lost` or `winner`. The `tasks` column shows the same N in both rows: both LoRAs and null M were judged on the same tasks.
4. A `lost` row has a note with the reason, most often `z ... is not above 1`. If N is below 8, the note also says `scored on N tasks; the rule needs 8`; that is still a pass.
5. In the `wall time` line with no winner, `total` is at most `25m00s`. With a winner, `lease to promotion` is at most `25m00s`; the fold comes after it and is not in the limit.
6. The JSON summary shows `"phase": "IDLE"`, and `served` is 2 more than before: both seats are charged.

Pass for `limit-not-scored`, all of these:

1. The first line is `SN120 v2 round r00000N: NO_DECISION (stopped at the time limit)`.
2. The line `not scored: 2 of 2 batches. The time limit of the round came before the first look. They are not charged and are first in the next round.`
3. Two rows with the verdict `not_scored` and the note `the round's time limit ended the scoring after 0 tasks, before the first look of 2000`.
4. In the `wall time` line, `total` is at most `8m00s`. One case is known: the stop time does not cut the start of a vLLM server. If the servers were still starting at minute 6, `total` is more than 8 minutes. Lines 1, 2, 3 and 5 are then still a pass of the rule; send the `wall time` line.
5. The JSON summary shows `"phase": "IDLE"`, and `last_outcomes` ends with `"no_decision"`. `served` and `heads` are the same as before the round: nobody is charged, and both seats are still in the queue. The next round takes these two seats first.

Not a pass, and what to do:

- `limit-judged` shows rows with `not_scored`: the first block did not finish before minute 23. That is not a fault of the rule. Send the `eval` lines of `phases.jsonl` and the `wall time` line; the time limit of this file must be higher. Nobody is charged.
- `limit-not-scored` shows rows with `lost` or `winner`: the box scored 2000 tasks in 6 minutes, or it did not stop. Send `run-limit-not-scored.log`.
- A row with `no_decision` and the note `this batch had the whole training time of the round ... and did not finish`: the training of a small copy took longer than the training time of the file. Send the `train` lines of `phases.jsonl`.
- In `limit-judged`, `total` or `lease to promotion` is above the time limit: the steps after the stop (the stop of the vLLM servers, the result, the promotion) took more than `promote_reserve_s`. Send the `wall time` line.

After these two rounds, go on with `5-full`: its `set-params` puts the launch values back.

Things to know:

- Whole files that are already in the smoke prefix (from an earlier step 5) are not valid under the cap of 2048 (rule `output_tokens`). They are passed over in stages 1, 2 and 4. That is correct. Under the launch cap they are valid again. So stage 3 can take some of them, and its `run` first checks all of them, which can take some minutes.
- A real file can fail the rules `no_signal` (no group has two different rewards) or `train_cost` (training it costs too much). So stage 5 can see fewer valid files than it copied. `python -m sn120_v2 status --state-dir $STATE --stack sn120_v2.assembly:stack` shows the count for each rule.
- The weights dry run of step 8 takes `--params`. Give it the file of the stage that is pinned now, or leave `--params` out. With another file it stops with `--params differs from the parameters pinned in the journal`.
- Do not edit a stage file by hand: the `smoke params` command writes it again.

## 5. Copy a few real queued batches into the smoke prefix

This reads v1's bucket with a read-only token and writes only under the smoke prefix. `limit=32` copies the files of the first 32 waiting hotkeys that have a file, in queue order.

```
export V1_R2_ENDPOINT=... V1_R2_BUCKET=... V1_R2_ACCESS_KEY_ID=... V1_R2_SECRET_ACCESS_KEY=...   # read-only token
python - <<'EOF' | tee $SMOKE/copied.txt
import os
from pathlib import Path
from sn120_v2.cutover import copy_miner_files, preview_queue
from sn120_v2.params import Params
from sn120_v2.r2 import R2Store
from sn120_v2.types import ChainSnapshot, V1Import

env, out = os.environ, Path(os.environ["SMOKE"]) / "cutover"
assert env.get("SN120_R2_PREFIX", "").startswith("smoke"), "set the smoke prefix: this copy never writes to v1's keys"
imp = V1Import.from_json((out / "v1_import.json").read_bytes())
snapshot = ChainSnapshot.from_json((out / "chain.json").read_bytes())
hotkeys = [seat.reg.hotkey for seat in preview_queue(imp, snapshot, Params())]
src = R2Store(endpoint=env["V1_R2_ENDPOINT"], bucket=env["V1_R2_BUCKET"],
              access_key_id=env["V1_R2_ACCESS_KEY_ID"], secret_access_key=env["V1_R2_SECRET_ACCESS_KEY"])
print("\n".join(copy_miner_files(src, R2Store.from_env(), hotkeys, limit=32)))
EOF
grep -c "default recipe" $SMOKE/copied.txt
```

Expected: one line for each copied file, `copied <hotkey> <bytes> bytes sha256=<hash> default recipe` or `... train={...}`. A file with no `train` object uses the default recipe.

Now count how many of the queued files are valid under the v2 rules. This runs the full admission check (format, tokens, grader) on every copied file, so it can take some minutes.

```
python -m sn120_v2 status --state-dir $STATE --stack sn120_v2.assembly:stack
```

Expected: the summary of step 4, plus `queued_files`, a count for each verdict rule. For example:

```
"queued_files": {"no file": 190, "valid": 21, "<name of a failed rule>": 11}
```

`no file` is a waiting seat with no file in the smoke prefix. Each other name is the rule that made the files invalid (for example a train key that only v1 had).

Pass: the `grep` count is at least 1 (at least one batch with the default recipe), and `valid` is at least 2. If no default-recipe file is valid, the test still runs, but step 9 then has no norm of the default recipe; say so in the report.

## 6. One full round on the 8 GPUs

```
python -m sn120_v2 run --state-dir $STATE --stack sn120_v2.assembly:stack --rounds 1 2>&1 | tee $SMOKE/run.log
```

What happens:

1. The controller reads the registrations from the chain, and gives the first 32 seats with a valid file a lease (the right to be trained and scored in this round). A seat with no valid file is skipped: it keeps its place and is not charged.
2. Training: 8 batches at a time, one GPU each, in queue order. One LoRA per batch. Training has its own time: 40 minutes from the lease (`train_time_budget_s`). At its end no new training starts, and a job that still runs is stopped.
3. Evaluation: one vLLM server per GPU. Null M and every LoRA answer the same tasks with the same seeds, at temperature 1. There are 3 looks (after 500, 1000 and 2000 tasks). At each look a LoRA that is clearly not better is dropped. The tasks are scored in blocks of 500 (`eval_chunk_tasks`).
4. Decision: the best LoRA wins if z > 1. Else there is no winner, and the head and the kings stay as they are.
5. Promotion and fold: the winner's LoRA is published, then added to the master.

The time limit of a round is 100 minutes from the lease to the promotion. The box stops scoring at minute 90, with no exception. The 10 minutes after it are for the grading of the last block, the judging and the promotion. The rules:

- A batch that was not trained when the training time ended has the verdict `not_trained`. It is not charged, it is first in the next round, and no failure is counted for its seat. So the scoring starts at minute 40 at the latest.
- The box starts no new block of tasks after minute 90. A block that still runs then is stopped and is not used.
- A LoRA that is still open at the limit is judged by z > 1 on the blocks that null M and every open LoRA finished.
- If fewer tasks than the first look (500) are scored at the limit, every trained LoRA has the verdict `not_scored`. It is not charged, it is first in the next round, and no failure is counted for its seat.
- A round ends early when every LoRA is dropped or decided.

The summary shows the time of each phase.

Watch it from a second shell (same exports):

```
python -m sn120_v2 status --state-dir $STATE          # "phase": "RUNNING", then "APPLY", then "IDLE"
tail -f $SN120_WORK_DIR/logs/phases.jsonl             # one line per train, eval and fold call, with wall times
tail -f $SN120_WORK_DIR/logs/vllm-0.log               # the vLLM server on GPU 0
nvidia-smi
```

Expected at the end: the round summary as text (see step 9), then the JSON summary with `"rounds_closed": 1`.

Pass, all of these:

1. The first line of the round summary is `SN120 v2 round r000001: APPLIED` or `... NULL`. Both are a pass. `NULL` means no LoRA beat null M by more than one standard error.
2. Every batch has a verdict: `winner` or `lost`. With 8 batches on 8 GPUs, expect no `not_trained` and no `not_scored`. `lease to promotion` in the `wall time` line is at most `100m00s`; if it is more, send the `wall time` line: the steps after the stop took more than 10 minutes.
3. The JSON summary shows `"phase": "IDLE"` and `"head_folded": true`.
4. With `APPLIED`: `"heads": 3` (genesis, the head with the winner's LoRA, the folded head), and `kings` starts with the winner's UID.
5. `nvidia-smi` shows the GPUs free after the command ends.
6. The journal replays from the store alone. This command uses a new empty directory, so it rebuilds the state from the journal copy in the store. It must print the same summary:

   ```
   python -m sn120_v2 status --state-dir $SMOKE/replay-check --stack sn120_v2.assembly:stack
   ```

Not a pass, and what to do:

- `NO_DECISION`: a failure of the box or the store, or no batch was trained inside the training time. Nobody is charged; the seats are back at the front. Read the `note` column and `run.log`, fix the cause, run step 6 again.
- Rows with `not_scored`: the first block of 500 tasks did not finish before minute 90. Nobody is charged, and the same seats are first in the next round, so the next round would end the same way. Send the `wall time` line and the `eval` lines of `phases.jsonl`; `train_time_budget_s`, `batch_cap` or the number of tasks must change.
- A row with `no_decision` and the note `this batch had the whole training time of the round ... and did not finish`: one batch on one GPU needs more than 40 minutes. Send the `train` lines of `phases.jsonl`; `train_time_budget_s` or the cost cap of a batch must change.
- `ABANDONED`: the winner's files did not match their hash. The head did not change. Stop and report it.
- `refused:` on the last line: a check failed closed. The line names the check.
- Lines `waiting: ...` that repeat: the chain or the store does not answer. If the chain read is the cause, set `SN120_V2_CHAIN_NETWORK=none` (the queue then stays as bootstrap made it) and run again.

If the box dies in the round: start a box, do steps 1 and 2 with the **same** `SN120_R2_PREFIX`, and run the step 6 command with an empty `$STATE`. It pulls the journal from the store and continues or closes the open round.

If a fix lands while a round is open, or a round cannot finish: see "After a fix".

## 7. Publish of `mining.json` and `llms.txt`

`bootstrap` published both before the round, and the controller publishes again when a round closes. This command publishes on demand; it reads the journal and does not take it over.

```
python -m sn120_v2 publish --state-dir $STATE --stack sn120_v2.assembly:stack
python - <<'EOF'
import json
from sn120_v2 import layout
from sn120_v2.r2 import R2Store
store = R2Store.from_env()
print(store.get(layout.LLMS_TXT).decode()[:1500])
mining = json.loads(store.get(layout.MINING_JSON))
print(sorted(mining))
print("current head:", json.loads(store.get(layout.CURRENT))["head"]["version_id"])
print("round files:", store.list("public/rounds/"))
print("left in pending/:", store.list("pending/"))
EOF
```

Expected: `{"published": true, "head": "...", "queue": <seats>}`, then the start of `llms.txt`, the top-level keys of `mining.json`, the head, the round files and an empty `pending/` list.

Pass: `llms.txt` states the LoRA recipe, the miner-settable bounds and the round rules; the head in the head pointer (`layout.CURRENT`) is the `head` of `status`; `public/rounds/r000001/` has `round.json` (the summary record) and, with a winner, `batch.json` (the winner's batch); `pending/` is empty.

## 8. Weights: dry run only

The dry run reads the kings from the journal and the registrations from the chain, and prints the table it would set. It needs no wallet and writes nothing. Never run this command without `--dry-run` in the smoke test.

```
python -m sn120_v2.weights_writer --state-dir $STATE --network finney --params $SMOKE/params.json --dry-run
```

Expected: one JSON line, `{"dry_run": true, "status": "would set", "kings": <count>, "weights": {"<uid>": <share>, ...}, "validator_hotkey": "5HmY...", "validator_registered": true, "block": <block>}`.

Pass: `status` is `would set`; `weights` has at most 10 UIDs with equal shares that sum to 1; UID 0 is not in `weights` (no burn); `validator_registered` is `true`. If the round had a winner, its UID is in the table.

The hand-off to AffineValidator. AffineValidator is the validator host: a separate machine that holds only the validator hotkey and the public file of the coldkey, and runs only the hourly weights writer. It is not the box. It has no journal of its own, so it reads the journal copy in the store. That read needs only a read-only token, and it never writes to the store.

Check that read path here, on the box, with a second empty directory. The first command pulls the journal copy into it. The second is the writer's dry run on that copy.

```
python - <<'EOF'
import os
from pathlib import Path
from sn120_v2.r2 import R2Store
from sn120_v2.state import load_from_store
state = load_from_store(R2Store.from_env(), Path(os.environ["SMOKE"]) / "weights-host" / "journal.jsonl")
print("kings, oldest first:", [king.uid for king in state.kings])
print("head:", state.head.version_id)
EOF
python -m sn120_v2.weights_writer --state-dir $SMOKE/weights-host --network finney --params $SMOKE/params.json --dry-run
```

Pass: the first command prints the kings and the head of `status`; the second prints the same `weights` as above.

## 9. The round summary, and the measured update norm

```
python -m sn120_v2 round --state-dir $STATE --stack sn120_v2.assembly:stack | tee $SMOKE/round.txt
python -m sn120_v2 round --state-dir $STATE --stack sn120_v2.assembly:stack --json > $SMOKE/round.json
```

The first command prints the text to paste. The second saves the full record, which is also in the store at `public/rounds/r000001/round.json`. Add `--round r00000N` for an older round. The shape of the text (the numbers here are from a simulated round):

```
SN120 v2 round r000001: APPLIED
winner: UID 1 (hk1), delta +0.3594, z 5.94
head: 11bd14fd7b69 -> bda0f8a1e929
batches: 3 | tasks scored: 64 of 64 | null M: pass 39.06%, cap-hit 0.00%
pos  uid  hotkey       verdict        delta      z cap-hit tasks      norm  note
  0    1  hk1          winner       +0.3594   5.94   0.00%    64      0.01
  1    2  hk2          lost         +0.0000   0.00   0.00%    64      0.01  z 0.000 is not above 1
  2    3  hk3          lost         -0.0625  -1.44   0.00%    32      0.01  dropped at look 0; scored on 32 tasks, not on the full draw; z -1.438 is not above 1
wall time: train 0m04s | eval 0m03s | promote 0m00s | fold 0m00s | lease to promotion 0m09s | total 0m09s
```

- `delta`: pass rate of the LoRA minus pass rate of null M, on the same tasks.
- `cap-hit`: the share of answers that stopped at the output token limit.
- `tasks`: the number of tasks this LoRA was scored on.
- `norm`: the update norm that training gave, before any cap.

The norm for `recipe.global_norm_cap`. Take the `norm` of a batch that step 5 listed as `default recipe`. This prints the norm and the wall time of every training of the round:

```
grep '"phase": "train"' $SN120_WORK_DIR/logs/phases.jsonl | python -c "
import json, sys
for row in map(json.loads, sys.stdin):
    print(row['queue_pos'], 'norm before cap', row['step_norm_before_cap'], 'rows', row['rows'], 'wall_s', round(row['wall_s']))"
```

The mainnet genesis sets `recipe.global_norm_cap` to about 4 times the norm of the default recipe. It goes into the parameter file of the mainnet bootstrap; no code changes:

```
{"recipe": {"image_digests": ["..."], "code_digest": "...", "global_norm_cap": <4 x the measured norm>}}
```

Pass: `round.txt` has one row for each batch with its delta, z, cap-hit rate and verdict, the winner (or `no winner`), and the wall time of each phase; at least one default-recipe batch has a norm.

## What to send back

1. The commit (`git rev-parse HEAD`) and `$SN120_R2_PREFIX`.
2. `$SMOKE/round.txt`.
3. The `queued_files` counts of step 5.
4. The norm of the default recipe (step 9).
5. The dry-run line of step 8.
6. Three things for the launch sizes of a round. Send them after the largest stage that you ran, and again from `5-full`:
   - The whole file `$SN120_WORK_DIR/logs/phases.jsonl`. It has one line for each training, each block of tasks and each fold, with the times and the token counts. In `5-full`, send it as soon as the first 500 tasks end (the first `"phase": "eval"` line of that round). Do not wait for the end of the round.
   - `$SMOKE/copied.txt`. Step 5 writes this file again, so send the earlier one before step 5.
   - The output of `grep "Avg generation throughput" $SN120_WORK_DIR/logs/vllm-*.log`. Send these lines only, with the file name in front, not the whole logs. Each vLLM server writes its speed there every 10 seconds.

   Keep the files in `$SN120_WORK_DIR/logs` between the stages.
7. Any step that did not pass, with its output.

## After a green test: the restart loop of `run`

On mainnet nothing but this loop starts `run` again after a crash or a reboot. So on mainnet do not start `run` alone: start the loop. It is one script, `systemd/sn120-v2-run-loop.sh`; its head lists every rule and setting. The unit `systemd/sn120-v2-run.service` starts the same script.

Before each start of `run` the loop does three things, in this order:

1. It stops every process that the last `run` left (vLLM servers, training jobs) and waits until no GPU holds more than 1,024 MiB. If a GPU stays in use for 120 seconds, the loop stops and says which GPUs.
2. It ends an open, undecided round with `end-round`: no decision, nobody is charged, no failure is counted, and the seats are first in the next round. If the box result of that round is already in the store, it does not end the round: `run` judges that result.
3. It starts `run`.

What a restart does to an open round:

| The round when `run` died | `run` alone, started by hand | The loop |
|---|---|---|
| In the training time (to minute 40) | All batches train again from zero. A batch of the first group that does not finish gets `no_decision`: **one counted failure** for that seat (3 hold the seat). Nobody is charged. | `end-round`: no decision, nobody charged, no failure counted. |
| After the training time, no result stored | The round closes with no decision. Nobody charged, no failure counted. The scores are lost. | The same, at once. |
| The box result is stored, not judged | `run` judges the stored result. Also after the lease end. | The same: the loop skips `end-round`. |
| Decided (`"phase": "APPLY"`) | `run` finishes the promotion and the fold. The charges were fixed with the decision. | The same. |

So a restart can never charge a seat. Only a restart of `run` by hand in the training time can count a failure. **When you restart `run` by hand with an open, undecided round, use the order of the loop: stop `run`, free the GPUs, `end-round`, start.** The loop does that order by itself.

Which exit codes of `run` the loop does not start again:

| Exit code of `run` | Meaning | The loop |
|---|---|---|
| 0 | Finished (`--rounds`, `--max-idle`) | stops, exit code 0 |
| 2 | Refused: a line `refused: <cause>` (parameters, state, a journal that another controller took) | stops, exit code 2. Only `refused: InfraError` (the store or the chain did not answer) is tried again. |
| 130, 143 | Stopped with SIGINT or SIGTERM | stops, exit code 0 |
| every other code | A crash (1 is a Python error, 137 is SIGKILL or out of memory) | starts `run` again after 30 seconds |

The loop counts the starts with no finished round between them. A round that `run` closed sets the count back to zero; a round that the loop ended with `end-round` does not. At 5 (`SN120_LOOP_MAX_STARTS`) the loop stops with exit code 3. A crash 30 minutes into every round stops it too. The last lines of `$SN120_WORK_DIR/logs/run.log` then say why, and what to run.

**A box without systemd** (a rented container). In `v2/`, in the shell of step 2:

```
setsid nohup systemd/sn120-v2-run-loop.sh "$(command -v python)" --state-dir $STATE --stack sn120_v2.assembly:stack > /dev/null 2>&1 < /dev/null &
tail -f $SN120_WORK_DIR/logs/run.log                  # the output of run, and the lines "run-loop: ..."
cat $SN120_WORK_DIR/logs/run-loop.status              # one line: what the loop does now, or why it stopped
kill $(cat $SN120_WORK_DIR/logs/run-loop.pid)         # the stop: run and its servers end, and nothing starts them again
```

After a reboot of such a box nothing starts the loop, unless the start command of the box holds the first line. If it does not, start the loop by hand.

**A box with systemd.** In `v2/`, in the shell of step 2:

```
RENV=$SMOKE/run.env
(umask 077; env | grep '^SN120_' > $RENV)             # the variables of run, mode 600. It holds the store token.
sed -e "/^[^#]/s|SN120_USER|$(id -un)|" -e "/^[^#]/s|SN120_ENV_FILE|$RENV|" -e "/^[^#]/s|SN120_PYTHON|$(command -v python)|" \
    -e "/^[^#]/s|SN120_V2_DIR|$PWD|" -e "/^[^#]/s|SN120_STATE_DIR|$STATE|" \
    systemd/sn120-v2-run.service | sudo tee /etc/systemd/system/sn120-v2-run.service > /dev/null
grep -v '^#' /etc/systemd/system/sn120-v2-run.service | grep SN120_      # must print nothing: every blank is filled
systemd-analyze verify /etc/systemd/system/sn120-v2-run.service
sudo systemctl daemon-reload && sudo systemctl enable --now sn120-v2-run
systemctl status sn120-v2-run                         # "active (running)"; after the loop gave up: "failed"
sudo systemctl stop sn120-v2-run                      # the stop. "sudo systemctl disable --now sn120-v2-run" also keeps it down after a reboot.
```

Stop the loop before `set-params`, before `end-round` by hand, and before a new commit. One box only: stop the loop on an old box (and `disable` its unit) before a new box starts.

One trial on this box, with the smoke state (it costs one round, which ends with no decision). Start the loop, wait for the training of a round (`nvidia-smi` shows used memory), then:

```
kill -9 $(pgrep -f "sn120_v2 run")
```

Pass: `run.log` gets the lines `run-loop: stopping what the last run left: pids ...`, `run-loop: end-round: the open round is ended with no decision.` and `run-loop: start 2 of at most 5 ...`; `status` shows `no_decision` as the last entry of `last_outcomes`; the next round runs to its end.

## After a fix

A fix is a new commit on `v2`. The state continues at the new commit: there is no new bootstrap. The message for the fix says if it needs the install line and if it needs the digest block.

First look at the state:

```
python -m sn120_v2 status --state-dir $STATE          # "phase": "IDLE", "RUNNING" or "APPLY"
```

Then:

```
git fetch origin && git checkout <the commit of the fix>
uv pip install -e ".[dev,gpu,r2,chain,envs,miner]"    # only when the fix needs the install line
```

The fix needs **no** digest block: start `run` again. In phase `RUNNING`, end the round first with the `end-round` line below (stop `run`, free the GPUs, `end-round`, start): a start of `run` in the training time of an open round can count a failure for a seat. The restart loop does this order by itself.

A fault in the judging of a round stops `run` with a Python error and keeps the round open, with the box result in the store. After the fix, `run` judges that stored result; no new round is needed. `end-round` leaves such a round alone and exits with 3. Only if the stored result itself is bad, `end-round --discard-result` ends the round with no decision and throws the result away; nobody is charged.

The fix needs the digest block (a package changed, or a file under `sn120_v2/gpu/` changed). What to do depends on the phase:

- `IDLE`: the digest block of step 1, the `smoke params` command, `set-params` with the file of the stage, then `run`.
- `RUNNING` (the round was not decided): the open round pins the old digests, so this box can no longer finish it. End it first. Nobody is charged, and its seats are first in the next round:

  ```
  python -m sn120_v2 end-round --state-dir $STATE --stack sn120_v2.assembly:stack
  ```

  Then do the `IDLE` line.
- `APPLY` (the round is decided; the promotion or the fold is left): do the `IDLE` line. In this phase `set-params` takes a file that differs from the pinned one in the two digests only, so use the file of the same stage. `run` then finishes the fold with the new code.

`end-round` is also the command for a round that cannot finish for any other cause, for example when the box is gone. Stop `run` first. It is refused when no round is open, and when the round is already decided.

The cheapest restart, when the state itself is lost or wrong: the block in the next section. It keeps the genesis files (no upload of the 56 GB), the miner files and the box cache. It removes the journal and everything the rounds wrote. Then do step 4 from the `bootstrap` command on.

## Start the test again from the beginning

The journal copy is in the store. So a new state directory with the same prefix continues the old test; it does not start a new one. For a new test, pick one:

- A new `SN120_R2_PREFIX` and a new `$STATE`, then steps 3 to 9. Step 3 uploads the 56 GB again.
- Or keep the prefix and delete everything but the genesis files and the copied miner files. This block refuses any prefix that does not start with `smoke`:

```
python - <<'EOF'
import os
from sn120_v2 import layout
from sn120_v2.r2 import R2Store
assert os.environ.get("SN120_R2_PREFIX", "").startswith("smoke"), "this block is only for a smoke prefix"
store = R2Store.from_env()
tops = ("private/journal/", "private/rounds/", "pending/", "public/heads/", "public/rounds/", "public/loras/", "public/irt/")
keys = [key for top in tops for key in store.list(top)] + [layout.CURRENT, layout.LLMS_TXT, layout.MINING_JSON]
for key in keys:
    if store.exists(key):
        store.delete(key)
print("deleted", len(keys), "keys under", os.environ["SN120_R2_PREFIX"])
EOF
rm -rf $STATE
```

Then do step 4 from the `bootstrap` command on.

## What only this test proves

The code passes its tests on CPU with a tiny model and fake vLLM servers. This run is the first with the real model and real GPUs. Each of these may fail here:

1. The real model loads for training, and the fast kernels are used.
2. vLLM 0.32.0 loads the LoRA files that the trainer writes, 32 of them on each server, and the 8 servers start side by side.
3. The same task and seed give the same sampling for null M and for every LoRA.
4. Memory and speed with long rows (up to 32,768 output tokens in training, up to 65,536 in generation).
5. The GPUs are free between training and evaluation.
6. The times: how long one batch trains on one GPU (against the 40 minutes of training time), how long a block of 500 tasks takes, and so how many batches and tasks fit in 100 minutes. If the tasks do not fit, the time limit ends the round and the summary says `(stopped at the time limit)`; if not even the first look fits, the trained LoRAs are `not_scored`.
7. Fold time and RAM with the real 37.6 GB master, and R2 speed for 56 GB.
8. How many real v1 files are valid under the v2 rules.
9. The update norm of the default recipe.
