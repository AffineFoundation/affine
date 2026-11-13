# Cutover: start v2 on mainnet from v1's state

This is the operator's runbook for the mainnet cutover. Do the smoke test first: [SMOKE.md](SMOKE.md). Cut over only after it passes.

## Who does what

- **Jacob (the operator)** does every step on a host, a wallet or the chain. No step is automatic.
- **The coordinator** archives the v1 state of `main` as the branch `v1-archive` and the tag `v1-final`. Then `v2` takes over `main`. Only the coordinator does this. The coordinator names the commit for the cutover.
- **No worker** touches a host, a wallet, the chain or `main`.

## Not in this file yet

Each point is a fact or a command that did not exist when this file was written. Do not guess it. Each place below that needs one says **Not landed yet**.

1. The preflight command (hard gate, "GPU box", step 5).
2. The head pointer at `public/head.json`. The code of this commit still writes `public/current.json`, which v1 owns ("Before the mainnet bootstrap", check 4).
3. The command of the public pull on AffineValidator, and the steps that reset the site ("AffineValidator", steps 11 and 12). The website worker writes them.
4. The setting on the GPU box that puts the address of the link files into `llms.txt` ("GPU box", step 2).
5. The numbers from the smoke test for `params.json`: `recipe.global_norm_cap`, and any size that the smoke test changes.
6. Unit files for `run`, the public pull and the site builder. The repository has unit files for the weights writer and the upload links job (`v2/systemd/`).

## Hosts

This file names no machine address. Jacob has them.

| Host | Runs | Holds | Never holds |
| --- | --- | --- | --- |
| **GPU box** (one secure 8x B300) | One process: `python -m sn120_v2 run`. It is the controller, the round runner, the folder and the publisher. | An R2 key with read and write. Read-only chain access. | A wallet. |
| **AffineValidator** (the validator host) | The website, and three v2 jobs: the hourly weights writer, the upload links job, the public pull. v1's weight writer and v1's `affine-ff.service` run here until the cutover. | The validator hotkey. The public file of its coldkey. Three credentials files, one for each job. | A state directory from the GPU box. |

- The GPU box is disposable. Its state directory is a cache of the journal. The durable copy of the journal is in R2 (`private/journal/`), one object for each record. A new box runs the same `run` command and continues from that copy ("A new GPU box").
- Each job on AffineValidator has its own credentials file and its own directory for its copy of the journal. No job shares a directory with another, and nothing is copied from the GPU box.

| Job on AffineValidator | Credentials file | Directory |
| --- | --- | --- |
| Weights writer | Its own env file with a read-only R2 token. The unit never reads another credentials file. | Its own, for example `$HOME/sn120-v2-weights` |
| Upload links job | v1's file `/home/const/subnet120-rewrite/state/private/r2-direct.env`, read in place (`--credentials-file`). It can sign an upload; v1's `affine-ff.service` signs with it. | Its own, for example `$HOME/sn120-v2-links` |
| Public pull | Its own env file with a read-only R2 token. | Its own |

## Production values (Jacob, 13 Nov 2026)

| Setting | Value |
| --- | --- |
| `SN120_R2_ENDPOINT` | `https://00523074f51300584834607253cae0fa.r2.cloudflarestorage.com` |
| `SN120_R2_BUCKET` | `affine-verification-mock-20260930` (v1's bucket) |
| `SN120_R2_PREFIX` | Empty. Leave the variable out, or set it to nothing. The smoke test has its own prefix in the same bucket. |
| `SN120_R2_ACCESS_KEY_ID`, `SN120_R2_SECRET_ACCESS_KEY` | From the secret store. Never in the repository. |
| `SN120_PUBLIC_URL` | `https://affine.io` |
| Site directory on AffineValidator | `/home/const/subnet120/affine/website` |
| Address of the link files (`--links-url`) | `https://affine.io/ff/links` |
| Address of `llms.txt` | `https://affine.io/llms.txt`, as in v1 |
| Validator hotkey (sets weights from v2, as in v1) | `5HmYnmUYT6qe3yFMg1Ad8WLLqDvnwtjYakXBpDvoRW1Qqzb8` |
| Its coldkey | `5Fc3ZZQAYB3SPXKcFnd1WJeyQvArSZZeB6LU1rb7zvQ6XvDh` |
| Subnet owner hotkey (`--owner-hotkey`) | The hotkey of UID 0. On 13 Nov 2026 the chain shows the validator hotkey at UID 0. The chain is the truth: storage item `SubnetOwnerHotkey` of subnet 120. |
| v1 turns with no verdict | To the front of the v2 queue, not charged (`--undecided front`, the default). On 13 Nov 2026 there are 14. The turn of epoch 1157 stays among them. |
| Burn | None, in any case. UID 0 has no seat and gets no weight. If no king is registered, the writer sets nothing and the last weights stay on chain. |
| `batch_cap` | 32, the default. The smoke test's 8 is not carried over. |
| Genesis model | `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`, fixed in the code. Until the first winner, the head is that model on Hugging Face. |

Nothing serves the bucket to the public. The public pull on AffineValidator copies v2's small public files into the site directory, and the website serves them.

`public/` in the bucket is not v2's own: v1 has objects there too. v2's keys are the ones that the table in "Store" names, and no step of this file acts on the whole of `public/`.

## The state of v1 (13 Nov 2026)

v1's evaluation was torn down at 4:55 PM Costa Rica time (22:55 UTC). So v1 starts no new turn, and the part "stop v1's rounds" is done. What still runs: v1's hourly weight writer, `affine-ff.service` (upload links) and the publishers of the v1 site. Jacob stops each of them at its step of the list "AffineValidator".

What the operator still checks, on the host where v1's round path ran (`systemctl --user`):

1. No unit of group A in "v1 units" is active or enabled. The watchdogs come first: a watchdog that still runs starts v1's round path again.
2. v1's last epoch does not move. Run the import dry run ("GPU box", step 4) two times, 10 minutes apart. `v1_last_epoch` in the `source:` line, the kings and the front seats must be the same. On 13 Nov 2026 at 23:04 UTC the dry run passed with `v1_last_epoch=1314`, 10 kings, 14 front seats and 256 waiting seats.

**Save `k1.json` now**, and again right before the import's real run. `k1.json` shows only v1's last 200 epochs, and it is the only record of a turn that v1 took and did not judge. Keep all copies.

```
curl -fsS https://affine.io/k1.json -o k1-$(date -u +%Y%m%dT%H%M%SZ).json
```

The first copy is in this repo: `tests/data/cutover/k1.json` (13 Nov 2026, 22:38 UTC).

**The import can run now.** It reads only public files and the chain. Do the dry run now. Do the real run on the cutover build, right before the bootstrap.

## What the import carries

`python -m sn120_v2.cutover` reads two public files, `https://affine.io/k1.json` and `https://affine.io/mining.json`. It needs no credential, reads no v1 host and no bucket, and writes nothing to the chain. It writes:

- `genesis.json` (`GenesisSpec`): the fresh MiMo genesis head and the validator hotkey.
- `v1_import.json` (`V1Import`): v1's last 10 kings; each registration and hotkey that had its turn in v1; the seats v1 moved to the back; the seats that are owed a turn at the front.
- `chain.json`: a copy of the chain snapshot that you gave.
- A copy of each v1 file that it read (the evidence).

At the start, v2 builds its queue from the chain: each registration that is not in the served lists waits, in order of registration block, then UID. Thus each paid registration gets one turn, and no registration gets two. The kings are v2's first kings, oldest first, so the first v2 weights pay the same 10 miners that v1 pays. A registration that is made after the import joins the queue when `run` reads the chain.

A queued miner file is not in the import. v2 reads it in place, at the same key as v1: `private/ff/v1/miners/<hotkey>/batch.json`. Nothing is copied. A v1 file with train keys that only v1 had is invalid in v2: the hotkey keeps its place and is not charged until the miner uploads a new file.

## Before the mainnet bootstrap: run these again on the box

The cutover build is not the commit of the smoke test. A result of the smoke test does not carry over by itself. On the GPU box:

1. Check out the commit that the coordinator names, and install again. The packages can differ.

   ```
   git fetch origin && git checkout <commit that the coordinator names>
   git rev-parse HEAD        # write this down with the cutover records
   cd v2 && source .venv/bin/activate
   uv pip install -e ".[dev,gpu,r2,chain,envs,miner]"
   ```

2. The test suite and the CPU check. Pass: `pytest` shows no failure; `sim` prints `"replay_matches": true`.

   ```
   python -m pytest
   python -m sn120_v2 sim --state-dir /tmp/sn120-sim-cutover --rounds 6 --miners 4
   ```

3. The digests. A new commit has a new code digest, and new packages give a new image digest. The digests of the smoke test are stale: with them, the first train call stops with an error.

   ```
   export CUT=<a local disk with 300 GB free>/sn120-mainnet
   mkdir -p $CUT
   python -m sn120_v2.gpu.plan | tee $CUT/plan.json
   export SN120_IMAGE_DIGEST=$(python -c "import json,os; print(json.load(open(os.environ['CUT']+'/plan.json'))['image_digest'])")
   nvidia-smi --query-gpu=index,name,memory.used --format=csv     # 8 GPUs, almost no memory in use
   ```

4. The head pointer. It must print `public/head.json`. If it prints `public/current.json`, this build would overwrite a v1 object: stop. (**Not landed yet** at the commit of this file.)

   ```
   python -c "from sn120_v2 import layout; print(layout.CURRENT)"
   ```

5. A new parameter file, `$CUT/params.json`. Do not use the smoke test's file: it holds the old digests and `batch_cap` 8. The coordinator names `global_norm_cap` (about 4 times the update norm that the smoke test measured, SMOKE.md step 9) and any other value that the smoke test changed.

   ```
   {"recipe": {"image_digests": ["<image_digest of plan.json>"], "code_digest": "<code_digest of plan.json>", "global_norm_cap": <number>}}
   ```

6. A new, empty state directory and the production store settings ("GPU box", step 2). Never use the state directory of the smoke test: `run` would copy the smoke journal into the production bucket.

On AffineValidator, check out the same commit and install again before the list "AffineValidator".

## GPU box: the steps

Run every command from `v2/`, inside the virtual environment.

1. **The checks of "Before the mainnet bootstrap"**, all six.
2. **Paths and store.**

   ```
   export STATE=$CUT/state                   # new and empty
   export SN120_WORK_DIR=$CUT/work           # model cache, job files, logs. Disposable.
   mkdir -p $SN120_WORK_DIR
   export SN120_R2_ENDPOINT=https://00523074f51300584834607253cae0fa.r2.cloudflarestorage.com
   export SN120_R2_BUCKET=affine-verification-mock-20260930
   export SN120_R2_ACCESS_KEY_ID=... SN120_R2_SECRET_ACCESS_KEY=...     # read and write; from the secret store
   unset SN120_R2_PREFIX                     # production has no prefix
   export SN120_PUBLIC_URL=https://affine.io
   export SN120_V2_CHAIN_NETWORK=finney      # read-only chain reads; the box has no wallet
   ```

   Check: `echo "[$SN120_R2_PREFIX]"` prints `[]`. Do not run `python -m sn120_v2.r2 --self-test` here: the smoke test proved the store, and with no prefix the self-test would write test keys at the top of v1's bucket.

   **Not landed yet:** the variable that gives the publisher the address of the link files (`https://affine.io/ff/links`). Without it, `llms.txt` says `Where to get the link: not published yet`.
3. **Chain snapshot.** It needs no wallet.

   ```
   python -m sn120_v2.chain --netuid 120 --network finney --out $CUT/chain_snapshot.json
   ```

4. **Import dry run.** It prints the import and writes nothing. Give `--earlier-k1` one time for each saved copy of `k1.json`.

   ```
   export HK=5HmYnmUYT6qe3yFMg1Ad8WLLqDvnwtjYakXBpDvoRW1Qqzb8
   python -m sn120_v2.cutover --dry-run --chain $CUT/chain_snapshot.json --owner-hotkey $HK \
       --earlier-k1 tests/data/cutover/k1.json --earlier-k1 <each later copy>
   ```

   Read the output:
   - The 10 kings are the miners that v1 pays now.
   - The front seats are the turns that v1 did not judge. Each line shows the v1 epoch of the turn, and how many turns that hotkey had in v1. The seat of epoch 1157 (UID 923) must be among them.
   - `first weights` must show 10 UIDs.
   - The last line of the report must say `UID 0 is <address>: it has no seat and gets no weight; it is the given owner hotkey`.
5. **Preflight (hard gate).** A read-only command lists every key and prefix that v2 writes, and fails when one of them already exists in the bucket. Run it before the first production write, which is step 6. If it fails, stop the cutover: v2 would write over an object that is not its own. **Not landed yet:** the command line. Do not go on without it.
6. **Genesis files** (57 GB of disk, 30 GB of RAM, no GPU). With the genesis directory of the smoke test, nothing is downloaded or built again; the files are uploaded to `public/bases/<base hash>/` and `public/masters/<master digest>/` of the production store, and each stored file is read back and compared.

   ```
   python -m sn120_v2.gpu.genesis --out <the genesis directory of the smoke test> --upload 2>&1 | tee $CUT/genesis.log
   export MASTER_DIGEST=$(tail -n 1 $CUT/genesis.log | python -c "import json,sys; print(json.load(sys.stdin)['master_digest'])")
   ```

   Check: `echo $MASTER_DIGEST` prints the same 64 characters as in the smoke test.
7. **Import, real run.** Save `k1.json` once more, take a new chain snapshot (step 3), and run:

   ```
   python -m sn120_v2.cutover --out $CUT/import --chain $CUT/chain_snapshot.json \
       --master-digest $MASTER_DIGEST --validator-hotkey $HK --owner-hotkey $HK \
       --earlier-k1 tests/data/cutover/k1.json --earlier-k1 <each later copy>
   ```

   Check: no `CUTOVER IMPORT REFUSED`; the kings and the front seats are those of step 4; one `WROTE` line for each file. The command never replaces a directory: for a second run, use a new one. If v1's publisher of `k1.json` has stopped, the command refuses a file older than one hour: add `--max-age-s` with the smallest value that accepts it.
8. **First start.** `--stack sn120_v2.assembly:stack` names the builder of the real parts. The command pins the parameters, writes the genesis head, the kings and the first queue, and publishes `llms.txt` and `mining.json` into the store. It refuses a second run on a journal that has a genesis.

   ```
   python -m sn120_v2 bootstrap --state-dir $STATE --params $CUT/params.json --stack sn120_v2.assembly:stack \
       --genesis $CUT/import/genesis.json --genesis-files <the genesis directory>/genesis_files.json \
       --v1-import $CUT/import/v1_import.json --chain $CUT/import/chain.json
   python -m sn120_v2 status --state-dir $STATE --stack sn120_v2.assembly:stack
   ```

   Check: `kings` holds the 10 UIDs of step 4; `queue` is the count of waiting seats of step 7; `published` is `true`; `queued_files` of `status` shows how many waiting seats have a valid file.
9. **Tell AffineValidator to start its list.** The journal copy is now in the store. No step of that list waits for step 10 here.
10. **Start the round loop.** One process. It runs until it is stopped. Start it under tmux or under your own unit.

    ```
    python -m sn120_v2 run --state-dir $STATE --stack sn120_v2.assembly:stack 2>&1 | tee -a $CUT/run.log
    ```

    Check, from a second shell with the same exports: `status` shows a `phase` other than `IDLE` after the first lease, and `python -m sn120_v2 round --state-dir $STATE --stack sn120_v2.assembly:stack` prints the summary when the first round has closed.
11. **Records.** Put `$CUT/import`, `$CUT/params.json`, `$CUT/plan.json` and each saved `k1.json` with the v2 run records.

## AffineValidator: the order at the cutover

One list, in this order. Each step has its check. Do not start a step before the check of the step before it passes. Steps 1 to 3 can be done before the cutover. Steps 4 to 13 need step 8 of "GPU box" (the bootstrap).

The detail of the weights writer is in the workers' document `api-weights.md`, and that of the links job in `api-links.md`; the commands here are the same.

Run every step as the Linux user that runs v1's `affine-ff.service`: that user can read v1's credentials file and owns the site directory. Names for this session (full paths, no spaces):

```
RUNUSER=$(whoami)
REPO=$HOME/affine                          # the checkout
WSTATE=$HOME/sn120-v2-weights              # the weights writer's own directory
WENV=$HOME/sn120-v2-weights.env            # the weights writer's own credentials file
WALLETS=$HOME/.bittensor/wallets           # the directory that holds the wallet NAME
WALLET=NAME/HOTKEY                         # the wallet name and the hotkey name of the validator hotkey
LSTATE=$HOME/sn120-v2-links                # the links job's own directory
CRED=/home/const/subnet120-rewrite/state/private/r2-direct.env     # v1's credentials file, read in place
SITE=/home/const/subnet120/affine/website
LINKS_URL=https://affine.io/ff/links
ENDPOINT=https://00523074f51300584834607253cae0fa.r2.cloudflarestorage.com
BUCKET=affine-verification-mock-20260930
KEEP=$HOME/sn120-v1-archive                # where files of v1 are kept
mkdir -p "$KEEP/site"
```

1. **Install.** The same commit as the GPU box.

   ```
   cd "$REPO" && git fetch origin && git checkout <commit that the coordinator names>
   cd v2 && uv venv .venv --python 3.12 && source .venv/bin/activate
   uv pip install -e ".[chain,r2,links]"
   install -d -m 700 "$WSTATE" "$LSTATE"
   ```

   Check: `python -c "import bittensor, boto3, nacl, sn120_v2.weights_writer, sn120_v2.links; print('ok')"` prints `ok`.
2. **The weights writer: credentials file and unit files.** Write `$WENV` with these five lines, with no quotes and no `export`, then `chmod 600 "$WENV"`. The token is the read-only one.

   ```
   SN120_R2_ENDPOINT=https://00523074f51300584834607253cae0fa.r2.cloudflarestorage.com
   SN120_R2_BUCKET=affine-verification-mock-20260930
   SN120_R2_ACCESS_KEY_ID=
   SN120_R2_SECRET_ACCESS_KEY=
   SN120_R2_PREFIX=
   ```

   Fill the blanks of the service file and install both units. Do not enable anything yet.

   ```
   sed -e "/^[^#]/s|SN120_USER|$RUNUSER|" -e "/^[^#]/s|SN120_ENV_FILE|$WENV|" \
       -e "/^[^#]/s|SN120_PYTHON|$REPO/v2/.venv/bin/python|" -e "/^[^#]/s|SN120_STATE_DIR|$WSTATE|" \
       -e "/^[^#]/s|SN120_NETWORK|finney|" -e "/^[^#]/s|SN120_WALLET_NAME|$WALLET|" \
       -e "/^[^#]/s|SN120_WALLETS_DIR|$WALLETS|" \
       "$REPO/v2/systemd/sn120-v2-weights.service" | sudo tee /etc/systemd/system/sn120-v2-weights.service > /dev/null
   sudo cp "$REPO/v2/systemd/sn120-v2-weights.timer" /etc/systemd/system/
   sudo systemctl daemon-reload
   ```

   Check: both commands print nothing.

   ```
   grep -v '^#' /etc/systemd/system/sn120-v2-weights.service | grep SN120_
   systemd-analyze verify /etc/systemd/system/sn120-v2-weights.service /etc/systemd/system/sn120-v2-weights.timer
   ```

3. **The links job: host check and unit files.** The job reads v1's credentials file in place. Never use `cat`, `grep` or `source` on that file.

   ```
   cut -s -d= -f1 "$CRED"                      # prints the names in the file, never a value
   stat -c '%U %a %n' "$CRED" "$SITE/ff" "$SITE/ff/links"
   sed -e "/^[^#]/s|LINKS_USER|$RUNUSER|" -e "/^[^#]/s|LINKS_R2_ENDPOINT|$ENDPOINT|" \
       -e "/^[^#]/s|LINKS_R2_BUCKET|$BUCKET|" -e "/^[^#]/s|LINKS_PYTHON|$REPO/v2/.venv/bin/python|" \
       -e "/^[^#]/s|LINKS_STATE_DIR|$LSTATE|" -e "/^[^#]/s|LINKS_CREDENTIALS_FILE|$CRED|" \
       -e "/^[^#]/s|LINKS_SITE_DIR|$SITE|" -e "/^[^#]/s|LINKS_PUBLIC_URL|$LINKS_URL|" \
       -e "/^[^#]/s|LINKS_NETWORK|finney|" \
       "$REPO/v2/systemd/sn120-v2-links.service" | sudo tee /etc/systemd/system/sn120-v2-links.service > /dev/null
   sudo cp "$REPO/v2/systemd/sn120-v2-links.timer" /etc/systemd/system/
   sudo systemctl daemon-reload
   ```

   Check: `cut` shows the names `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`. `stat` shows `$RUNUSER` as the owner of each path. Both commands below print nothing. Do not enable anything yet.

   ```
   grep -v '^#' /etc/systemd/system/sn120-v2-links.service | grep LINKS_
   systemd-analyze verify /etc/systemd/system/sn120-v2-links.service /etc/systemd/system/sn120-v2-links.timer
   ```

4. **Dry run of the weights writer, with no wallet.** It pulls the journal copy with the read-only token and writes nothing to the chain.

   ```
   cd "$REPO/v2" && source .venv/bin/activate
   ( set -a; . "$WENV"; set +a
     python -m sn120_v2.weights_writer --state-dir "$WSTATE" --from-store --network finney --dry-run )
   ```

   Check: stderr says `kings: read from the journal copy in the store`. The JSON line has `"status": "would set"`; `weights` holds the 10 UIDs of the import at 0.1 each, and no UID 0; `validator_hotkey` is the hotkey above; `"validator_registered": true`. The parentheses keep the token out of the shell that the later steps use.
5. **Dry run with the wallet.** It also reads the hotkey file and runs every check that comes before a submission. Still no chain write.

   ```
   ( set -a; . "$WENV"; set +a
     python -m sn120_v2.weights_writer --state-dir "$WSTATE" --from-store --network finney \
         --wallet "$WALLET" --wallet-path "$WALLETS" --dry-run )
   ```

   Check: `wallet_hotkey` equals `validator_hotkey`, and either `"preflight": "passed"`, or `"status": "would defer"` with the reason `recent write: the rate limit allows the next one in N blocks`. That reason is normal while v1's writer still writes. Any other reason, or `nothing submitted: the hotkey file ...`: stop and correct it.
6. **Rehearsal of the links job into an empty directory.** It writes nothing to the store and nothing to the website.

   ```
   REH=$(mktemp -d)
   env -u SN120_R2_PREFIX SN120_R2_ENDPOINT=$ENDPOINT SN120_R2_BUCKET=$BUCKET \
     "$REPO/v2/.venv/bin/python" -m sn120_v2.links issue --state-dir "$LSTATE" --credentials-file "$CRED" \
     --site-dir "$REH" --links-url "$LINKS_URL" --network finney
   ls "$REH/ff/links"  | sort > /tmp/v2-links.txt
   ls "$SITE/ff/links" | grep '\.json$' | sort > /tmp/v1-links.txt
   comm -12 /tmp/v1-links.txt /tmp/v2-links.txt | wc -l    # link files that the first run replaces
   comm -23 /tmp/v1-links.txt /tmp/v2-links.txt | wc -l    # link files that the first run removes
   comm -13 /tmp/v1-links.txt /tmp/v2-links.txt | wc -l    # link files that are new
   rm -r "$REH"
   ```

   Check: one line `links: issued N of M waiting hotkeys (not activated ..., not Ed25519 ...), removed 0, valid 86400 s`, exit code 0. `M` is the queue of the bootstrap. If the second count is most of v1's files, stop and ask: then the v2 queue is not what the import promised.
7. **Move the weights writer from v1 to v2. This is one step: do not pause inside it.** Two writers on one hotkey overwrite each other. The chain allows one write per 100 blocks for one hotkey (20 minutes), and v1's writer writes each hour (300 blocks). So do this step when `blocks_since_last_write` of step 4 is between 100 and 250: v1's last write is then old enough for v2's first write, and v1's next write has not come yet.
   1. Stop and disable v1's hourly weight writer ("v1 units", group B). Check: `systemctl --user list-timers --all` shows no v1 writer, and the `grep` of group B finds no enabled unit. `affine-hourly-burn.timer` and `affine-transition-weights.timer` stay masked.
   2. One run of v2's writer, at once. This is the first real weight write of v2. A write waits about one minute for the block to be final.

      ```
      sudo systemctl start sn120-v2-weights.service
      journalctl -u sn120-v2-weights.service -n 20 --no-pager
      ```

      Check: the last line is JSON with `"status":"set"`. With `"status":"deferred"`, nothing was written and the reason is in the line; v1's last weights stay on chain meanwhile. Each clock hour has one attempt; for a second one in the same hour, correct the cause and run by hand:

      ```
      ( set -a; . "$WENV"; set +a
        python -m sn120_v2.weights_writer --state-dir "$WSTATE" --from-store --network finney \
            --wallet "$WALLET" --wallet-path "$WALLETS" --force )
      ```

   3. The timer: one run each hour at minute 5, also after a reboot. Enable the timer, never the service.

      ```
      sudo systemctl enable --now sn120-v2-weights.timer
      systemctl list-timers sn120-v2-weights.timer
      ```

      Check: the timer is listed with its next run. After the next hour, `python -m json.tool "$WSTATE/weights_state.json"` shows `last_receipt.status` `set`.
8. **Stop v1's upload links and v1's publishers, and move `ff/status.json` away.** Stop and disable `affine-ff.service`, `affine-k1.service` and the other units of group C in "v1 units". `ff/status.json` is written by `affine-ff.service`: after the stop it would show v1's last view for ever, and v2 has no file of this kind.

   ```
   systemctl --user disable --now affine-ff.service        # and each other unit of group C
   mv "$SITE/ff/status.json" "$KEEP/ff-status-v1.json"
   ```

   Check: `systemctl --user is-active affine-ff.service` prints `inactive`, and `is-enabled` prints `disabled`. `ls "$SITE/ff/status.json"` finds no file, and `curl -s -o /dev/null -w '%{http_code}\n' https://affine.io/ff/status.json` prints `404` (a cache can serve the old file for a short time).
9. **Save v1's site files and freeze the v1 history.** Do it before the first run of the pull and of the site builder: the pull writes `llms.txt` and `mining.json` at the root of the site, over v1's files, and the builder writes the nine files of the `for` line.

   ```
   for f in mining.json k1.json llms.txt scores.json benchmarks.json \
            index.html v1.html site.css site.js favicon.svg network-haffer.ttf network-mono.ttf site-data.json v1-data.json; do
     [ -e "$SITE/$f" ] && cp -a "$SITE/$f" "$KEEP/site/"; done
   python site/v1_archive.py --from "$SITE" --out "$KEEP/v1-history.json"
   ```

   Check: `ls "$KEEP/site"` shows at least `mining.json`, `k1.json` and `llms.txt`. The second command prints `v1 archive: wrote ... 10 kings, N turns, M benchmark rows`; it reads `k1.json`, `k1/turns/index.json`, `scores.json` and `benchmarks.json`, and writes nothing unless all four are readable. Keep the file: it cannot be made again. On 13 Nov 2026 the same command on `https://affine.io` gave 10 kings, 857 turns and 261 benchmark rows.
10. **First run of the links job, then its timer.** Two conditions: `affine-ff.service` is stopped (step 8), and the bootstrap is done. See "Upload links job" for the reason.

    ```
    sudo systemctl start sn120-v2-links.service
    journalctl -u sn120-v2-links.service -n 5 --no-pager
    curl -s https://affine.io/ff/links.json | head -c 400
    sudo systemctl enable --now sn120-v2-links.timer
    systemctl list-timers sn120-v2-links.timer --no-pager
    ```

    Check: the log shows the `links: issued ...` line of step 6; `removed` is now the second count of step 6. In `ff/links.json`, `issued_at` is new and `link_file` is `https://affine.io/ff/links/<hotkey>.json`. One link opens with a test hotkey that you own: `python -m sn120_v2.links open --hotkey <address> --key <key file> --links-url https://affine.io/ff/links`. The timer is listed (minute 10 and minute 40 of each hour).
11. **Start the public pull.** With its own read-only token and its own directory, it writes `llms.txt` and `mining.json` at the root of the site, and `public/read_links.json`: signed read links for the weight files, new on every run. It does not copy a weight file, and it takes only keys that v2 writes ("Store"), not everything under `public/`. **Not landed yet:** the command.

    Check: `curl -fsS https://affine.io/llms.txt | head -n 5` shows the v2 text. `curl -fsS https://affine.io/mining.json | head -c 300` shows `snapshot_digest` and the v2 head. `curl -fsS https://affine.io/public/read_links.json | head -c 300` answers.
12. **Build the v2 site.** **Not landed yet:** the steps that reset the site, and the value of `--source`. The builder of this commit reads `<source>/public/mining.json`; the form is:

    ```
    python site/build.py --source <the root that holds the v2 snapshot> --public-url https://affine.io --out "$SITE" \
        --v1-history "$KEEP/v1-history.json" --every 60
    ```

    The builder writes its seven static files, `site-data.json` and `v1-data.json`, and nothing else; it leaves `ff/links/` alone. With `--every 60` it builds the data again each minute, and a failed read keeps the last good `site-data.json`. Until a benchmark file exists it says one time `site: no benchmark table`.

    Check: `https://affine.io` shows the v2 page with the 10 kings and the queue, and `https://affine.io/v1.html` shows the v1 history.
13. **Read the public files as a miner does.** `https://affine.io/llms.txt` names `https://affine.io/ff/links/<hotkey>.json` in the line `Where to get the link`. If that line says `not published yet`, tell the coordinator before the announcement.

## After the start

On the GPU box, in the shell of step 2 of "GPU box".

| Command | What it does |
| --- | --- |
| `python -m sn120_v2 status --state-dir $STATE` | The summary from the local journal: head, kings, queue, phase, `params_hash`. Read-only. |
| `python -m sn120_v2 status --state-dir $STATE --stack sn120_v2.assembly:stack` | The same, read through the store, plus `queued_files`: the count of queued files for each admission result. |
| `python -m sn120_v2 round --state-dir $STATE --stack sn120_v2.assembly:stack` | The summary of the last closed round, as a text to paste. `--round r000007` picks one; `--json` prints the stored record. |
| `python -m sn120_v2 publish --state-dir $STATE --stack sn120_v2.assembly:stack` | Writes `llms.txt` and `mining.json` again. It does not change the journal. |
| `python -m sn120_v2 set-params --state-dir $STATE --stack sn120_v2.assembly:stack --params <file>` | Pins new parameters and prints the old and the new `params_hash`. Stop `run` first: this command takes over the journal, and a `run` that still runs stops at its next write. Then start `run` again. The file holds the whole set of overrides, not only the change. |

`--params` on `run`, `status` and `publish` is only a check: a file that differs from the pinned parameters is refused.

On AffineValidator: `journalctl -u sn120-v2-weights.service --since "3 hours ago" --no-pager` shows the last runs. The dry run of step 4 of "AffineValidator" shows `blocks_since_last_write`; one hour is 300 blocks, so it stays below about 310 while each hourly run sets. If the first log line of a run says `read from the cache of the journal copy`, the store did not answer or the store settings are wrong: the writer then pays the last kings that it read.

### A new GPU box

1. Stop `run` on the old box, if the box still answers.
2. On the new box: the checkout, the install, step 2 of "GPU box" with the same store settings and a new empty state directory, and `SN120_IMAGE_DIGEST` from `python -m sn120_v2.gpu.plan`.
3. If the new box gives other digests than the pinned parameters hold, pin them first with `set-params`.
4. Start `run`. It pulls the journal copy and continues. Do not run `bootstrap` again.

Two `run` processes never both write: the newer one takes over the journal, and the older one stops with `FencedError` at its next write. Do not let a supervisor start the old one again.

### Go back

Before the first v2 round ends:

1. `sudo systemctl disable --now sn120-v2-weights.timer sn120-v2-links.timer`. Stop `run`, the pull and the site builder.
2. Put the files of `$KEEP/site` back into the site directory, and `ff-status-v1.json` back as `ff/status.json`.
3. Start v1's weight writer, `affine-ff.service` and v1's publishers again. `affine-ff.service` writes the link files again at its next run.

The import only read public files, and v2 writes no key that v1 wrote, so v1's state in the bucket did not change. The objects that v2 wrote stay in the bucket under v2's own keys.

## Upload links job

A miner uploads its file through a link that only its hotkey can open. The job writes one sealed link file for each waiting hotkey that made the one-time activation on chain, and an index. It reads the queue from the journal copy in the store. One run takes about 30 seconds; the timer starts a run at minute 10 and minute 40 of each hour, and a link works for 24 hours.

- **`--credentials-file`** names v1's credentials file. The job reads the two parts of the token from it on each run (`R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`, v1's names). Nothing copies the secret, and systemd never loads the file. The token must be able to write: a link signed with a read-only token is refused at the upload. The bucket and the endpoint are `Environment=` lines of the unit; the unit has no prefix line.
- **`--state-dir`** is a directory of the job's own. It is not the directory of the weights writer or of the pull.
- **`--site-dir`** writes `ff/links/<hotkey>.json` and `ff/links.json` under the site directory. These are the paths and the format of v1, so the address of a miner's link file does not change. The job removes each other `.json` file in `ff/links/`: the link file of a hotkey that has no seat in v2.
- **`--network finney`** reads the activations. When the chain cannot be read, the run fails and writes nothing; the old links work for up to 23 hours more.

A failed run prints `issue failed: <reason>`, writes nothing and removes nothing. After a replacement of the token every link stops at once: start one run, then test one upload from a machine outside the host.

**Why `affine-ff.service` must be stopped before the first run.** The v1 service and this job write the same files in the same directory. Every 30 minutes v1 writes `ff/links/<hotkey>.json` and `ff/links.json`, and removes the file of each hotkey that is not eligible in v1's view. v1's view of the queue is frozen since the teardown. There is no lock between the two. If both run:

1. v1 removes the link files of hotkeys that only v2 knows, and v2 removes those of hotkeys with no seat in v2. A miner gets HTTP 404 at random.
2. The index `ff/links.json` changes between two lists.
3. v1 writes `ff/status.json` again after the runbook moved it away.

A link that v1 issued before the stop works until its own end, 24 hours at most. It points to the same object that v2 reads (the same bucket, no prefix, the same key), so a file uploaded through it reaches v2. The activations are on chain: nothing is copied.

Never run two link jobs that write to the same place. Never run `links activate --execute` as the operator: it is the miner's own chain write. Never paste an opened link into a log or a chat. Never copy the credentials file, and never show its content.

## v1 units

The names come from documents in this repository. The repository does not show which of them run now. The host is the truth: `systemctl --user list-units 'affine-*' 'sn120-*' '*watchdog*'` and `systemctl --user list-timers --all`.

**Group A: the round path. Torn down on 13 Nov 2026; check that each is inactive and disabled.**

1. The watchdogs: `tp-watchdog`, `ff-watchdog`, and any other unit that `systemctl --user list-units '*watchdog*'` shows. A watchdog reverts v1 or starts it again when the lifecycle is down for 90 seconds (`ops/free_file/README.md`, `ops/ff_train_params/README.md`).
2. `affine-mimo-lifecycle.service`: selects the next turn and starts the training.
3. `affine-promotion-gate.service`: judges a turn. With it stopped, the set of turns with no verdict does not change.
4. `affine-mimo-trainer-supervised.service` and `affine-mimo-evaluator-supervised.service`, with their `-tunnel` units (`ops/pods/roles.json`).

**Group B: the weights writer ("AffineValidator", step 7).**

- The repository names the hourly writer in two ways. `docs/MIMO_DEPLOYMENT_RUNBOOK.md` names `affine-live-reward-writer.timer` and `affine-live-reward-writer.service`. The K1 service (`ops/k1_pass_rates/services/k1_service.py`) says that the hourly writer is `weights_flip.py`, which reads `website/scores.json`; no document gives the unit name of that script.
- So find the writer on the host: `systemctl --user list-timers --all`, and `grep -rlE 'weights_flip|live_reward_writer|current_assessment_writer|set_weights' ~/.config/systemd/user/`. Stop and disable each timer and service that these show.
- `affine-hourly-burn.timer` and `affine-transition-weights.timer` are old writers (`ops/HOURLY_BURN.md`, `ops/EQUAL_REGISTRATION_WEIGHTS.md`). They must stay disabled and masked.

**Group C: upload links, publishers and side services ("AffineValidator", step 8).** Never run two writers of one public file.

- `affine-ff.service`: issues the miners' upload links, checks their files, writes `ff/status.json`.
- `affine-k1.service`: writes `website/scores.json` and `website/k1.json`. It does not write to the chain. The import refuses a `k1.json` older than one hour, so stop it after the import's real run, or use `--max-age-s`.
- `affine-mining-discovery.service` (`mining.json`), `affine-k1-transparency.service`, `affine-training-params.service`.
- `affine-prior-table.service`, `affine-perf-eval.service`, `affine-perf-eval-2.service`, `affine-perf-watchdog.timer`, `affine-bench.service`.
- `affine-site-baseline.timer`, `affine-site-k1-tables.timer`, `affine-site-slices.timer`, `affine-site-metrics.service`.

Leave `sn120-disk-guard.timer` on. `sn120-pod-watch.timer` only reports the state of the v1 pods and never rents (`ops/pods/README.md`); stop it when the v1 pods are released. Do not run `replace_pod.py replace` for a v1 role after the cutover.

## Store

v2 and v1 share one bucket, and v2 has no prefix. The rule: **no v2 write and no v2 delete reaches a v1 object.** The preflight ("GPU box", step 5) checks it on the bucket before the first write.

No top-level prefix belongs to v2 alone, except `pending/`. v1 has objects under `public/` and under `private/`. So v2's keys are exactly the ones below, and nothing lists, copies or deletes a whole `public/` or `private/`.

**What v2 writes** (`sn120_v2/layout.py`):

| Key or prefix | Writer |
| --- | --- |
| `public/head.json` (the pointer to the current head), `public/heads/` | controller |
| `public/llms.txt`, `public/mining.json` | publisher |
| `public/rounds/`, `public/loras/`, `public/irt/` | controller |
| `public/bases/`, `public/masters/` | the genesis upload, then the controller after each fold |
| `private/journal/`, `private/rounds/` | controller, admission |
| `pending/` | the box, for every result before the controller has checked it |
| `public/benchmarks.json` | the benchmark service, when it exists |

**What v2 reads and never writes:** `private/ff/v1/miners/<hotkey>/batch.json`, the miner's own file. The key is the same as in v1, on purpose. Only the miner writes it, through its upload link.

**What v2 deletes.** Only objects that v2 wrote itself:

- everything under a round's `pending/` prefix, when the round closes;
- a round's frozen copy of a miner file, `private/rounds/<round>/batches/<hotkey>.json`: when the same round is tried again and the miner's file is now missing or invalid, or when the copy does not hash to the checked bytes;
- the winner's public batch, `public/rounds/<round>/batch.json`, when the copy does not hash to the frozen bytes;
- a genesis file under `public/bases/` or `public/masters/` whose stored bytes differ from the local file (the check after the upload);
- any object whose write the store accepted and whose check then failed (`R2Store`): the object that was just written, at the key that was just written.

The upload links job writes and deletes in the site directory, not in the bucket: `ff/links/<hotkey>.json` and `ff/links.json`. It removes the link files of hotkeys with no seat in v2; those files were written by v1's `affine-ff.service`, and the job takes these two paths over on purpose.

**What v2 never writes and never deletes:** `mimo/` (v1's heads are under `mimo/affine-mimo9b-grpo-20261013/checkpoints/<ckpt>/`), `public/current.json`, `public/<epoch>/current.json` and `public/streams/<name>/current.json` (v1's pointers), `public/checkpoints/`, `public/epoch-evidence/`, and everything under `private/ff/v1/` (the miner files, and v1's turn records under `private/ff/v1/turns/`).

The list of v1's keys comes from v1's source and the public files. Nobody listed the bucket for it: the preflight is the check on the real bucket.

## Options of the import command

| Option | Meaning |
| --- | --- |
| `--dry-run` or `--out <dir>` | Exactly one. Print only, or write the records to a new directory. |
| `--master-digest` | sha256 digest of the genesis fp32 master files ("GPU box", step 6). No default. |
| `--validator-hotkey` | Public SS58 address of the validator hotkey. A wrong character is refused. No default. |
| `--undecided front` or `served` | For turns that v1 took and did not judge. `front` (default, Jacob's rule): not charged, first in the v2 queue. `served`: charged, no new turn. |
| `--earlier-k1 <file>` | A saved earlier copy of `k1.json`. Give it one time for each copy. A turn with no verdict that only an earlier copy shows gets the `--undecided` rule. If a newer copy shows the verdict of a turn, the turn stays judged. |
| `--chain <file>` | A `ChainSnapshot` as JSON ("GPU box", step 3). Adds checks, and prints the start queue, the first weights and the hotkey of UID 0. `--out` copies it as `chain.json`. |
| `--owner-hotkey` | The subnet owner hotkey. The command refuses when UID 0 of `--chain` has another hotkey. Needs `--chain`. |
| `--max-age-s` | Oldest `k1.json` accepted, in seconds. Default 3600. Earlier copies have no age limit. Use a larger value only if v1's publisher stopped, and then the smallest value that accepts the file. |
| `--k1`, `--mining` | Another URL or a local file. The defaults are the two public URLs. |

## When the import command refuses

It exits with code 2, prints `CUTOVER IMPORT REFUSED` and the reason, and writes nothing. It refuses when:

- a v1 file cannot be read, has an unknown version or lacks a field;
- `k1.json` is older than `--max-age-s`, or `mining.json` is from another epoch or shows a head that `k1.json` does not know;
- the served lists, their counts and the turn history do not agree, or a waiting seat is also listed as served;
- the payees are not the last 10 distinct GREEN turns (a `pending` or timeout verdict is never a king);
- an earlier copy is newer than `k1.json`, or shows a turn that is not in v1's served turns;
- a retry grant names a hotkey that had no turn;
- with `--chain`: the snapshot is older than v1's chain read, UID 0 is absent, the validator hotkey is not registered, or no king is registered (v2 never burns, so its writer would set no weights);
- with `--owner-hotkey`: UID 0 has another hotkey, or `--chain` is absent.

Do not edit a v1 file to make the command pass. Find the cause in v1, or ask.
