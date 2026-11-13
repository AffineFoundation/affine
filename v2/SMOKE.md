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
params = {"recipe": {"image_digests": [plan["image_digest"]], "code_digest": plan["code_digest"]}}
json.dump(params, open(os.environ["SMOKE"] + "/params.json", "w"), indent=1)
print(json.dumps(params, indent=1))
EOF
```

Pass: `plan.json` is one line, `{"code_digest": "<64 hex>", "image_digest": "<64 hex>"}`, and `params.json` holds the same two values.

`params.json` changes nothing else. So the round has the launch size: at most 32 batches, 2000 tasks, at most 100 minutes. It sets no `global_norm_cap`: the smoke test runs with no cap and measures the norm.

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
2. Training: 8 batches at a time, one GPU each. One LoRA per batch.
3. Evaluation: one vLLM server per GPU. Null M and every LoRA answer the same tasks with the same seeds, at temperature 1. There are 3 looks (after 500, 1000 and 2000 tasks). At each look a LoRA that is clearly not better is dropped.
4. Decision: the best LoRA wins if z > 1. Else there is no winner, and the head and the kings stay as they are.
5. Promotion and fold: the winner's LoRA is published, then added to the master.

The time limit: at most 100 minutes from the lease to the promotion. The box starts no new training and no new block of tasks after minute 90. A round ends early when every LoRA is dropped or decided. At the limit, a LoRA that is still open is judged by z > 1 on the blocks of tasks that null M and every open LoRA finished. A LoRA with fewer scored tasks than the first look is not charged and keeps its place.

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
2. `lease to promotion` in the `wall time` line is at most `100m00s`.
3. The JSON summary shows `"phase": "IDLE"` and `"head_folded": true`.
4. With `APPLIED`: `"heads": 3` (genesis, the head with the winner's LoRA, the folded head), and `kings` starts with the winner's UID.
5. `nvidia-smi` shows the GPUs free after the command ends.
6. The journal replays from the store alone. This command uses a new empty directory, so it rebuilds the state from the journal copy in the store. It must print the same summary:

   ```
   python -m sn120_v2 status --state-dir $SMOKE/replay-check --stack sn120_v2.assembly:stack
   ```

Not a pass, and what to do:

- `NO_DECISION`: a failure of the box or the store, or the time limit came before the first look. Nobody is charged; the seats are back at the front. Read the `note` column and `run.log`, fix the cause, run step 6 again.
- `ABANDONED`: the winner's files did not match their hash. The head did not change. Stop and report it.
- `refused:` on the last line: a check failed closed. The line names the check.
- Lines `waiting: ...` that repeat: the chain or the store does not answer. If the chain read is the cause, set `SN120_V2_CHAIN_NETWORK=none` (the queue then stays as bootstrap made it) and run again.

If the box dies in the round: start a box, do steps 1 and 2 with the **same** `SN120_R2_PREFIX`, and run the step 6 command with an empty `$STATE`. It pulls the journal from the store and continues or closes the open round.

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

Pass: `llms.txt` states the LoRA recipe, the miner-settable bounds and the round rules; the head in `current.json` is the `head` of `status`; `public/rounds/r000001/` has `round.json` (the summary record) and, with a winner, `batch.json` (the winner's batch); `pending/` is empty.

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
6. The `eval` lines of `$SN120_WORK_DIR/logs/phases.jsonl` (the wall time of each block of tasks).
7. Any step that did not pass, with its output.

## Start the test again from the beginning

The journal copy is in the store. So a new state directory with the same prefix continues the old test; it does not start a new one. For a new test, pick one:

- A new `SN120_R2_PREFIX` and a new `$STATE`, then steps 3 to 9. Step 3 uploads the 56 GB again.
- Or keep the prefix and delete everything but the genesis files and the copied miner files. This block refuses any prefix that does not start with `smoke`:

```
python - <<'EOF'
import os
from sn120_v2.r2 import R2Store
assert os.environ.get("SN120_R2_PREFIX", "").startswith("smoke"), "this block is only for a smoke prefix"
store = R2Store.from_env()
tops = ("private/journal/", "private/rounds/", "pending/", "public/heads/", "public/rounds/", "public/loras/", "public/irt/")
keys = [key for top in tops for key in store.list(top)] + ["public/current.json", "public/llms.txt", "public/mining.json"]
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
6. A launch-size round (32 batches, 2000 tasks) fits in 100 minutes. If it does not, the time limit ends it and the summary says `(stopped at the time limit)`.
7. Fold time and RAM with the real 37.6 GB master, and R2 speed for 56 GB.
8. How many real v1 files are valid under the v2 rules.
9. The update norm of the default recipe.
