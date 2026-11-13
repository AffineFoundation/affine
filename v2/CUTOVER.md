# Cutover: start v2 on mainnet from v1's state

This is the operator's runbook for the mainnet cutover. Do the smoke test first: [SMOKE.md](SMOKE.md). Cut over only after it passes.

## Who does what

- **Jacob (the operator)** does every step on a host, a wallet or the chain. No step is automatic.
- **The coordinator** archives the v1 state of `main` as the branch `v1-archive` and the tag `v1-final`. Then `v2` takes over `main`. Only the coordinator does this. The coordinator names the commit for the cutover.
- **No worker** touches a host, a wallet, the chain or `main`.

## Not in this file yet

Each point is a fact or a command that is not on `v2` at the commit of this file. Do not guess it. Each place below that needs one says **Not landed yet**.

1. The preflight command (hard gate, "GPU box", step 5).
2. The head pointer at `public/head.json`. The code of this commit still writes `public/current.json`, which v1 owns ("Before the mainnet bootstrap", check 4).
3. The numbers from the smoke test for `params.json`: `recipe.global_norm_cap`, and any size that the smoke test changes.
4. A unit file for `run` on the GPU box. The three jobs on AffineValidator have unit files in `v2/systemd/`.
5. The pull's unit reads the names `R2_ENDPOINT`, `R2_BUCKET`, `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` of the pull's credentials file ("AffineValidator", step 12). The pull of this commit reads only the `SN120_R2_` names. The commands by hand do not wait for it: the helper `with_store` gives both sets of names.
6. The round time hard stop: the smallest commit whose `llms.txt` may go up, and the text check of the store's `llms.txt` ("AffineValidator", step 8).
7. The section "After a fix" of SMOKE.md. "After a fix: a new commit on the GPU box" here holds the same order; it takes that section's wording when it is on `v2`.

## Hosts

This file names no machine address. Jacob has them.

| Host | Runs | Holds | Never holds |
| --- | --- | --- | --- |
| **GPU box** (one secure 8x B300) | One process: `python -m sn120_v2 run`. It is the controller, the round runner, the folder and the publisher. | An R2 key with read and write. Read-only chain access. | A wallet. |
| **AffineValidator** (the validator host) | The website, and three v2 jobs: the hourly weights writer, the upload links job, the public pull. v1's weight writer and v1's `affine-ff.service` run here until the cutover. | The validator hotkey. The public file of its coldkey. Three credentials files, one for each job. | A state directory from the GPU box. |

- The GPU box is disposable. Its state directory is a cache of the journal. The durable copy of the journal is in R2 (`private/journal/`), one object for each record. A new box runs the same `run` command and continues from that copy ("A new GPU box").
- Each job on AffineValidator has its own credentials file. The weights writer and the links job each have their own directory for their copy of the journal. No job shares a directory with another, and nothing is copied from the GPU box.

| Job on AffineValidator | Credentials file | Directory |
| --- | --- | --- |
| Weights writer | Its own env file with a read-only R2 token. The unit never reads another credentials file. | Its own, for example `$HOME/sn120-v2-weights` |
| Upload links job | v1's file `/home/const/subnet120-rewrite/state/private/r2-direct.env`, read in place (`--credentials-file`). It can sign an upload; v1's `affine-ff.service` signs with it. | Its own, for example `$HOME/sn120-v2-links` |
| Public pull | Its own file, which exists already: `/home/const/sn120-v2/secrets/r2-public-pull.env`, mode 600. Its token can only read, and only this one bucket (Jacob verified: list and get work, a PUT is denied). It is not the credentials file of the upload links job. | None: it reads no journal. It writes only in the site directory. |

## Production values (Jacob, 13 Nov 2026)

| Setting | Value |
| --- | --- |
| `SN120_R2_ENDPOINT` | `https://00523074f51300584834607253cae0fa.r2.cloudflarestorage.com` |
| `SN120_R2_BUCKET` | `affine-verification-mock-20260930` (v1's bucket) |
| `SN120_R2_PREFIX` | Empty. Leave the variable out, or set it to nothing. The smoke test has its own prefix in the same bucket. |
| `SN120_R2_ACCESS_KEY_ID`, `SN120_R2_SECRET_ACCESS_KEY` | From the secret store. Never in the repository. |
| `SN120_PUBLIC_URL` | `https://affine.io` |
| `SN120_LINKS_URL` (GPU box only) | `https://affine.io/ff/links`. The publisher puts it into `llms.txt` and `mining.json`. |
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

2. The test suite and the CPU check. Pass: `pytest` shows no failure; `sim` prints `"replay_matches": true`. Run them in a new shell that has no `SN120_` variable (`env | grep ^SN120_` prints nothing): with `SN120_R2_PREFIX` of the smoke test still set, 5 tests of `tests/test_site_pull.py` fail.

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

4. Three changes that the build must hold. Stop if one of them is wrong. (The first and the third are **not landed yet** at the commit of this file; the second is.)

   ```
   python -c "from sn120_v2 import layout; print(layout.CURRENT)"
   grep -c SN120_LINKS_URL sn120_v2/assembly.py
   python -m sn120_v2.r2 --help | grep -c preflight
   ```

   - The first line must print `public/head.json`. With `public/current.json`, this build would overwrite a v1 object.
   - The second must print a count of 1 or more. With 0, this build does not read `SN120_LINKS_URL`, and `llms.txt` does not name the link files.
   - The third must print a count of 1 or more. With 0, this build has no preflight command.

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
   export SN120_LINKS_URL=https://affine.io/ff/links     # the address of the link files, for llms.txt
   export SN120_V2_CHAIN_NETWORK=finney      # read-only chain reads; the box has no wallet
   ```

   Check: `echo "[$SN120_R2_PREFIX]"` prints `[]`. Do not run `python -m sn120_v2.r2 --self-test` here: the smoke test proved the store, and with no prefix the self-test would write test keys at the top of v1's bucket.

   Without `SN120_LINKS_URL`, `llms.txt` says `Where to get the link: not published yet`, and a miner cannot find its link. Steps 8 and 13 of "AffineValidator" check the published text.
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
5. **Preflight (hard gate).** A read-only command lists every key and prefix that v2 writes, and fails when one of them already exists in the bucket. Run it before the first production write, which is step 6. If it fails, stop the cutover: v2 would write over an object that is not its own. Do not go on without it.

   **Not landed yet** at the commit of this file (PR #131). Nobody has run it against the real bucket: this run is the first. It only lists; it writes nothing.

   ```
   python -m sn120_v2.r2 --preflight
   ```

   Check: exit code 0 (`echo $?`), and the last line is `PASS  no object at a key that v2 writes`. The first line names the bucket and the key prefix: the bucket is the production one and the prefix is `''`. Each key or prefix of v2 has one line `ok`; the line for `private/ff/v1/miners` shows the count of miner files, which v2 reads in place.

   A line `EXISTS  <key or prefix>  N objects` and the last line `FAIL  objects exist at keys that v2 writes: do not start v2 on this bucket and prefix`: stop, and send the output to the coordinator. Do not delete anything to make it pass.

   After step 6, a second run finds v2's own genesis files. Then name them as expected, one `--allow` for each: `python -m sn120_v2.r2 --preflight --allow public/bases/ --allow public/masters/`.
6. **Genesis files** (57 GB of disk, 30 GB of RAM, no GPU). With the genesis directory of the smoke test, nothing is downloaded or built again; the files are uploaded to `public/bases/<base hash>/` and `public/masters/<master digest>/` of the production store, and each stored file is read back and compared.

   ```
   python -m sn120_v2.gpu.genesis --out <the genesis directory of the smoke test> --upload 2>&1 | tee $CUT/genesis.log
   export MASTER_DIGEST=$(tail -n 1 $CUT/genesis.log | python -c "import json,sys; print(json.load(sys.stdin)['master_digest'])")
   ```

   Check: `echo $MASTER_DIGEST` prints the same 64 characters as in the smoke test.
7. **Import, real run.** Before it, step 4 of "AffineValidator" is done: v1's site is saved. The import reads `https://affine.io/k1.json` and `https://affine.io/mining.json`, and the pull replaces `mining.json` later. Save `k1.json` once more, take a new chain snapshot (step 3), and run:

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
9. **Tell AffineValidator to go on with step 5 of its list.** The journal copy is now in the store. No step of that list waits for step 10 here.
10. **Start the round loop.** One process. It runs until it is stopped. Start it under tmux or under your own unit.

    ```
    python -m sn120_v2 run --state-dir $STATE --stack sn120_v2.assembly:stack 2>&1 | tee -a $CUT/run.log
    ```

    Check, from a second shell with the same exports: `status` shows a `phase` other than `IDLE` after the first lease, and `python -m sn120_v2 round --state-dir $STATE --stack sn120_v2.assembly:stack` prints the summary when the first round has closed.
11. **Records.** Put `$CUT/import`, `$CUT/params.json`, `$CUT/plan.json` and each saved `k1.json` with the v2 run records.

## AffineValidator: the order at the cutover

One list, in this order. Each step has its check. Do not start a step before the check of the step before it passes. Steps 1 to 4 can be done before the cutover; step 4 must be done before step 7 of "GPU box" (the import's real run). Steps 5 to 13 need step 8 of "GPU box" (the bootstrap).

The order follows Jacob's sequence: the new `llms.txt` first (step 8), then the weights and the stop of v1 (steps 9 and 10), then the website (step 12). The first run of the pull is the site switch, so the pull does not run before step 12.

The pull's credentials file exists already; step 8 checks it.

The workers' documents hold more detail on each job: `api-weights.md`, `api-links.md` and `api-site.md`. The commands here are the same.

Run every step as the Linux user that runs v1's `affine-ff.service`: that user can read v1's credentials file and owns the site directory. The weights writer runs as the same user, so that user must be able to read the hotkey file. Names for this session (full paths, no spaces):

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
PENV=/home/const/sn120-v2/secrets/r2-public-pull.env     # the pull's own credentials file; it exists already
KEEP=$HOME/sn120-v1-archive                # where files of v1 are kept, outside the site directory
BACKUP=$KEEP/website-v1                    # the copy of v1's whole site directory
V1FILE=$KEEP/v1-history.json               # the frozen v1 history
MOVED=$KEEP/moved                          # where v1's stale files go after the site switch
PY=$REPO/v2/.venv/bin/python               # the Python of the checkout
mkdir -p "$KEEP"
```

The site commands read the store settings from the names `SN120_R2_ENDPOINT`, `SN120_R2_BUCKET`, `SN120_R2_ACCESS_KEY_ID` and `SN120_R2_SECRET_ACCESS_KEY`. The pull's credentials file has the names without `SN120_`. Run this block once; it defines a helper named `with_store` that accepts both. The round brackets keep the token out of your shell: the helper runs each command in a shell of its own.

```
with_store() (
  set -a; . "$PENV"; set +a
  export SN120_R2_ENDPOINT="${SN120_R2_ENDPOINT:-$R2_ENDPOINT}" SN120_R2_BUCKET="${SN120_R2_BUCKET:-$R2_BUCKET}" \
         SN120_R2_ACCESS_KEY_ID="${SN120_R2_ACCESS_KEY_ID:-$R2_ACCESS_KEY_ID}" \
         SN120_R2_SECRET_ACCESS_KEY="${SN120_R2_SECRET_ACCESS_KEY:-$R2_SECRET_ACCESS_KEY}"
  "$@"
)
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

4. **Check the host, copy v1's site, and freeze the v1 history.** Do it before step 7 of "GPU box" (the import's real run). Nothing here changes the site directory.

   First the checks of the host. They only read.

   ```
   stat -c '%U %G %a %n' "$SITE" "$SITE/index.html" "$SITE/ff" "$SITE/ff/links"
   ls -d "$SITE/public"
   df --output=target "$SITE" "$KEEP"
   du -sh "$SITE"; df -h "$KEEP"
   curl -s -o /dev/null -w '%{http_code} %{content_type}\n' https://affine.io/ff/links/no-such-file.json
   ps -o user= -p "$(systemctl --user show -p MainPID --value affine-ff.service)"
   systemctl --user list-units 'affine-*' --no-pager; systemctl --user list-timers --all --no-pager
   ```

   Check, line by line:

   - `stat` shows `$RUNUSER` as the owner of each of the four paths.
   - `ls` says `No such file or directory`. If v1 has a `public/` directory, tell the coordinator before step 12: the pull refuses to replace a file there that it did not write.
   - The first `df` prints the same mount point two times. If the two differ, set `MOVED` to a directory on the mount point of `$SITE`, outside `$SITE`: the reset tool renames files and never copies them across disks.
   - The disk of `$KEEP` has more free space than the size of `$SITE`.
   - `curl` prints `404 application/json`. If it prints `200`, the web server answers with a page for a missing file: tell the coordinator, because a miner's tool then reads a page as if it were a link file.
   - `ps` prints the user that runs v1's `affine-ff.service`. It must be `$RUNUSER`. The v2 links job must run as the same user, or it cannot replace v1's link files.
   - Keep the list of units. Step 10 uses it.

   Then the copy and the frozen history. `$BACKUP` must not exist yet.

   ```
   cp -a "$SITE" "$BACKUP"
   "$PY" -c '
   import json, pathlib, sys
   n = bad = 0
   for p in pathlib.Path(sys.argv[1]).rglob("*.json"):
       n += 1
       try: json.loads(p.read_bytes())
       except ValueError: bad += 1; print("broken:", p)
   print(n, "json files,", bad, "broken")' "$BACKUP"
   ls -la "$BACKUP/k1.json" "$BACKUP/mining.json" "$BACKUP/llms.txt" "$BACKUP/index.html"
   grep -c 'site.js?v=' "$BACKUP/index.html"
   "$PY" "$REPO/v2/site/v1_archive.py" --from "$BACKUP" --out "$V1FILE"
   ```

   - `cp -a` copies everything and keeps owners, modes and times. The copy is the source of the way back.
   - `v1_archive.py` reads four files of the copy: `k1.json`, `k1/turns/index.json`, `scores.json` and `benchmarks.json`. It writes nothing unless all four are readable. The v1 history page of the v2 site reads only this frozen file.
   - The copy is also the last home of v1's `k1.json` and `mining.json`: the import can read them from a file (`--k1 "$BACKUP/k1.json"`, `--mining "$BACKUP/mining.json"`).

   Check:

   - The second command ends with `0 broken`. v1's writers still run, so one file can be caught in the middle of a write. For a broken file, copy that one file again with `cp -p`.
   - `ls` shows the four files.
   - `grep` prints `0`: the copy holds v1's page, not v2's.
   - The last command prints `v1 archive: wrote ... 10 kings, N turns, M benchmark rows`. On 13 Nov 2026 a copy gave 10 kings, 856 turns and 261 benchmark rows.
5. **Dry run of the weights writer, with no wallet.** It pulls the journal copy with the read-only token and writes nothing to the chain.

   ```
   cd "$REPO/v2" && source .venv/bin/activate
   ( set -a; . "$WENV"; set +a
     python -m sn120_v2.weights_writer --state-dir "$WSTATE" --from-store --network finney --dry-run )
   ```

   Check: stderr says `kings: read from the journal copy in the store`. The JSON line has `"status": "would set"`; `weights` holds the 10 UIDs of the import at 0.1 each, and no UID 0; `validator_hotkey` is the hotkey above; `"validator_registered": true`. The parentheses keep the token out of the shell that the later steps use.
6. **Dry run with the wallet, and the chain check.** The dry run also reads the hotkey file and runs every check that comes before a submission. Still no chain write.

   ```
   ( set -a; . "$WENV"; set +a
     python -m sn120_v2.weights_writer --state-dir "$WSTATE" --from-store --network finney \
         --wallet "$WALLET" --wallet-path "$WALLETS" --dry-run )
   ```

   Check: `wallet_hotkey` equals `validator_hotkey`, and either `"preflight": "passed"`, or `"status": "would defer"` with the reason `recent write: the rate limit allows the next one in N blocks`. That reason is normal while v1's writer still writes. Any other reason, or `nothing submitted: the hotkey file ...`: stop and correct it.

   Save the chain check script: the block in "The chain check" below. It only reads the chain and needs no wallet. Step 9 runs it several times. Then run it:

   ```
   python "$HOME/sn120-v2-chain-check.py"
   ```

   Check: it prints `commit-reveal on: True` and `rate limit in blocks: 100`. On 13 Nov 2026 at 23:09 UTC, while v1's writer ran, it printed:

   ```
   block: 9500636 | validator UID: 0
   commit-reveal on: True | reveal period in epochs: 1
   rate limit in blocks: 100 | last write block: 9500593 | next write allowed in blocks: 57
   next epoch in blocks: 111
   commit that waits for its reveal: block 9500593 | opens at 2026-11-13 23:32:27+00:00
   weights on chain: {821: 0.1, 822: 0.1, 851: 0.1, 852: 0.1, 909: 0.1, 918: 0.1, 920: 0.1, 942: 0.1, 955: 0.1, 956: 0.1}
   ```

   If it prints `commit-reveal on: False`, stop and tell the coordinator: that path was not tested. A larger rate limit is not a fault: the writer reads it from the chain, and the wait of step 9 is longer.
7. **Rehearsal of the links job into an empty directory.** It writes nothing to the store and nothing to the website.

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
8. **The new `llms.txt` alone.** Only this one file goes up here: the `llms.txt` that the bootstrap published into the store. v1's `mining.json`, `index.html` and every other file of the site stay as they are. Do not run the pull for this: one run of the pull also replaces `mining.json` and `index.html`.

   1. **Check the pull's credentials file.** It exists already: `$PENV`. Do not make a new one, do not edit it, and do not show its content.

      ```
      stat -c '%a %U' "$PENV"          # 600 and the run user
      cut -d= -f1 "$PENV"              # the names only, no value
      ```

      Check: the mode is `600`. The names are `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET` and `R2_ENDPOINT`. The token can only read, and only this one bucket. It is not the credentials file of the upload links job: that one can sign uploads.
   2. **Check that the store's `llms.txt` comes from a build with the round time hard stop.** `llms.txt` must not go up from a store that an older commit published. **Not landed yet:** the coordinator gives the smallest commit and the text to look for in the store's `llms.txt`. Do not go on before you have both.

      - The commit of the GPU box (`git rev-parse HEAD` there, check 1 of "Before the mainnet bootstrap") is that commit or a later one.
      - The text check of the store's `llms.txt`: the command comes with the commit.
   3. **Stop the v1 unit that puts v1's `llms.txt` back.** `affine-network-dashboard.service` writes `network-data.json` into the site directory every 15 seconds. In the same pass it compares `llms.txt` and `reward-policy.json` with its own copies and writes its copies back if they differ. While it runs, the new `llms.txt` lives for 15 seconds at most.

      ```
      systemctl --user disable --now affine-network-dashboard.service
      ```

      After this, v1's page shows `network-data.json` as it was at the stop. This is the cost of the new `llms.txt` going first.

      Check: `systemctl --user is-active affine-network-dashboard.service` prints `inactive`. If no such unit exists, find the process: `pgrep -af dashboard.server`.
   4. **Copy the file.**

      ```
      with_store "$PY" "$REPO/v2/site/reset.py" llms --site-dir "$SITE"; echo "exit=$?"
      ```

      - The command reads two objects from the store: v2's `llms.txt` and v2's `mining.json`. It writes one file: `$SITE/llms.txt`, in one move.
      - It copies only when both are of one snapshot. This is the pull's own rule. So the pull later finds the same bytes and writes nothing.
      - You can run it again at any time. If the GPU box published a newer snapshot, it puts the newer `llms.txt` up.

      Check:

      ```
      sha256sum "$SITE/llms.txt"
      curl -fsS https://affine.io/llms.txt | head -3
      curl -fsS https://affine.io/llms.txt | sha256sum
      curl -fsS -H 'Accept: text/html' https://affine.io/llms.txt | grep -o '<title>[^<]*</title>'
      curl -fsS https://affine.io/ | grep -c 'site.js?v='
      curl -fsS https://affine.io/llms.txt | grep 'Where to get the link'
      sleep 60; sha256sum "$SITE/llms.txt"
      ```

      - The command printed `reset llms: ... is the store's public/llms.txt: N bytes, sha256 H` and `exit=0`.
      - Both `sha256sum` lines and the `curl ... | sha256sum` line show `H`.
      - `head` shows `# Affine (Bittensor SN120) v2: the miner contract`.
      - The title line shows a page title: the browser view works.
      - `grep -c` prints `0`: the root still shows v1's page.
      - The `Where to get the link` line names `https://affine.io/ff/links/<your hotkey>.json`. If it says `not published yet`, see step 13.
      - After 60 seconds the hash is still `H`. If it changed, another v1 process writes `llms.txt`: v1's source speaks of "legacy website pushes". Find it (`pm2 ls`, `crontab -l`, `grep -rl llms ~/.config/systemd/user/`), stop it, and run the command again.

      With `exit=1`, the file in the site directory did not change. `missing in the store` means that the GPU box has not published yet, or that the bucket is wrong. `not of the same snapshot` means that a publish was in progress: run it again.

   To take only this step back (the new `llms.txt` is up, the site is still v1's): `cp -p "$BACKUP/llms.txt" "$SITE/llms.txt.tmp" && mv "$SITE/llms.txt.tmp" "$SITE/llms.txt"`, then start `affine-network-dashboard.service` again.
9. **Move the weights writer from v1 to v2. This is one step: do not pause inside it.** Only one writer can set weights: two writers on one hotkey block each other at the rate limit, and each overwrites the other. While v1's writer runs, every v2 run defers at the rate limit, so v1 stops first.

   **The best time: between minute 25 and minute 55 of an hour.** v1 writes at minute 01. The chain allows one write per 100 blocks for one hotkey (20 minutes), so from minute 21 the next write is allowed. Before minute 55, v1's next run is still far.

   SN120 has commit-reveal on: the weights go to the chain hidden (the commit), and the chain opens them by itself after one epoch, at most 72 minutes later (the reveal). So `set` means that the chain accepted the commit; the weights change at the reveal.

   1. **Check before the stop.** Run the chain check and the dry run with the wallet (step 6) again.

      Check: write down `last write block`; this is v1's last write. The `weights` of the dry run hold the 10 UIDs of the import, and before v2's first winner they are the UIDs of `weights on chain`; if the two sets differ, stop and ask the coordinator. The dry run gives `"preflight": "passed"` or the reason `recent write`.
   2. **Stop v1's writer.** "v1 units", group B, gives the candidate names; the host is the truth. For the timer and the service that the first two commands show:

      ```
      systemctl --user list-timers --all
      grep -rlE 'weights_flip|live_reward_writer|current_assessment_writer|set_weights' ~/.config/systemd/user/
      systemctl --user is-active <v1 writer>.service     # must print "inactive": do not stop a write that is in progress
      systemctl --user disable --now <v1 writer>.timer
      systemctl --user stop <v1 writer>.service
      ```

      Check: `systemctl --user list-timers --all` does not list the v1 writer's timer, and `systemctl --user is-enabled <v1 writer>.timer` prints `disabled`. `affine-hourly-burn.timer` and `affine-transition-weights.timer` stay disabled and masked. Never start them.
   3. **Wait for the rate limit.** Run the chain check again.

      Check: `last write block` is still the block of 9.1. If it moved, a v1 writer is still alive: go back to 9.2. Wait until it prints `next write allowed in blocks: 0`. Do not start v2 before this: a first run inside the rate limit submits nothing, and it uses the attempt of the hour.
   4. **First run of v2's writer.** This is the first real weight write of v2. The command returns when the run ends; a write waits about one minute for the block to be final.

      ```
      sudo systemctl start sn120-v2-weights.service
      journalctl -u sn120-v2-weights.service -n 20 --no-pager
      ```

      Check: the log shows `kings: read from the journal copy in the store`, then one JSON line with `"status":"set"`. With `"status":"deferred"`, nothing was submitted and the reason is in the line; v1's last weights stay on chain. Correct the cause, then run one more attempt by hand. `--force` only skips the rule of one attempt for each clock hour; every other check stays.

      ```
      ( set -a; . "$WENV"; set +a
        python -m sn120_v2.weights_writer --state-dir "$WSTATE" --from-store --network finney \
            --wallet "$WALLET" --wallet-path "$WALLETS" --force )
      ```

   5. **Enable the timer.** One run each hour at minute 5, also after a reboot. Enable the timer, never the service.

      ```
      sudo systemctl enable --now sn120-v2-weights.timer
      systemctl list-timers sn120-v2-weights.timer --no-pager
      ```

      Check: the timer is listed with its next run. If the first run and the first timer run are in the same clock hour, the timer run prints `this hour already had its attempt`; this is normal.
   6. **Check the commit, then the reveal.** Run the chain check at once.

      Check: `last write block` is a few blocks after the `block` of the `set` line, and a line `commit that waits for its reveal: block <the same block> | opens at <time>` is there. About 3 minutes after that time, run the chain check again: the `commit that waits` line is gone, and `weights on chain` equals the `weights` of the dry run.
   7. **Check the first timer run**, after the next minute 05: `journalctl -u sn120-v2-weights.service --since "70 minutes ago" --no-pager`.

      Check: the timer's run shows `"status":"set"`. A run that shows `recent write` means that another writer wrote in the last 20 minutes: find that writer and stop it (9.2).

   Go on with step 10 after the check of 9.5. The checks of 9.6 and 9.7 come later by the clock; do them when their time comes.
10. **Stop v1's writers of the site directory, then check that nothing writes.** v1's weight writer is already stopped (step 9). Stop and disable `affine-ff.service`, `affine-k1.service` and each other unit of group C in "v1 units" that the list of step 4 shows as active. Two writers of one public file are never allowed: a v1 unit that still runs writes `mining.json` or `llms.txt` over v2's files, and the pull writes them back a minute later, so the public files change back and forth.

    ```
    systemctl --user disable --now affine-ff.service affine-k1.service        # and each other unit of group C that is active
    touch "$KEEP/mark"; sleep 90
    find "$SITE" -newer "$KEEP/mark" -type f -not -path "$SITE/ff/links/*" -not -name links.json -not -path "$SITE/public/*"
    ```

    Check: for each unit, `systemctl --user is-active <unit>` prints `inactive` and `is-enabled` prints `disabled`. `issued_at` in `https://affine.io/ff/links.json` does not change any more. `find` prints nothing. Each line that it prints is a file that some process still writes: find that process and stop it, then run the last two lines again.
11. **First run of the links job, then its timer.** Two conditions: `affine-ff.service` is stopped (step 10), and the bootstrap is done. See "Upload links job" for the reason.

    ```
    sudo systemctl start sn120-v2-links.service
    journalctl -u sn120-v2-links.service -n 5 --no-pager
    curl -s https://affine.io/ff/links.json | head -c 400
    sudo systemctl enable --now sn120-v2-links.timer
    systemctl list-timers sn120-v2-links.timer --no-pager
    ```

    Check: the log shows the `links: issued ...` line of step 7; `removed` is now the second count of step 7. In `ff/links.json`, `issued_at` is new and `link_file` is `https://affine.io/ff/links/<hotkey>.json`. One link opens with a test hotkey that you own: `python -m sn120_v2.links open --hotkey <address> --key <key file> --links-url https://affine.io/ff/links`. The timer is listed (minute 10 and minute 40 of each hour).
12. **The website: the first run of the pull is the site switch.** One run copies v2's public files from the store into the site directory, writes `public/read_links.json` (signed read links for the weight files of the current head, new on every run), and builds the status site. It never writes to the store, it never copies a weight file, and it takes only keys that v2's code names, never everything under `public/`. See "Public pull and site".

    Once the store holds `public/mining.json` and `public/llms.txt`, a run replaces these files at the root of the site, with no check of who wrote them: `llms.txt`, `mining.json`, `index.html`, `v1.html`, `site.js`, `site.css`, `favicon.svg`, `network-haffer.ttf`, `network-mono.ttf`, `site-data.json`, `v1-data.json`. So no run of the pull, by hand or by the timer, starts before all of these:

    - step 4: the copy and the frozen history exist;
    - step 7 of "GPU box": the import has read v1's `mining.json` from `https://affine.io/mining.json`;
    - step 10: nothing of v1 writes into the site directory.

    The order inside this step: the pull first, then v1's files move out. Before the pull, v1's page still needs those files, and the reset tool refuses to move them until the v2 page is up.

    1. **One run of the pull by hand.**

       ```
       with_store "$PY" "$REPO/v2/site/pull.py" --site-dir "$SITE" --public-url https://affine.io --v1-history "$V1FILE"; echo "exit=$?"
       ```

       - The run writes the data files first (`mining.json`, `llms.txt`, the files under `public/`), then the fonts, the script and the style, and `index.html` and `v1.html` last. So no page is up before the files that it loads.
       - Do not run `build.py` by hand for the switch. Without the pull's data it writes a page that says "No data yet."

       Check on the host: the run prints `pull: wrote N, unchanged N, removed N, failed steps 0; site built` and `exit=0`. With `exit=1`, each failed step is one line `pull: FAILED ...` on stderr; before the GPU box has published, no root file was replaced, v1's page is still up, and you can run it again.

       Check from outside the host:

       ```
       for path in / /v1.html /site-data.json /v1-data.json /llms.txt /mining.json /public/mining.json /public/llms.txt \
                   /public/read_links.json /ff/links.json /robots.txt; do
         curl -s -o /dev/null -w "%{http_code} %{content_type} $path\n" "https://affine.io$path"
       done
       for asset in $(curl -s https://affine.io/ | grep -oE '[a-z0-9.-]+\?v=[0-9a-f]+' | sort -u); do
         curl -s -o /dev/null -w "%{http_code} /$asset\n" "https://affine.io/$asset"
       done
       curl -s -o /dev/null -w '%{http_code}\n' "https://affine.io/ff/links/$(ls "$SITE/ff/links" | head -1)"   # this one line on the host
       curl -fsS -H 'Accept: text/html' https://affine.io/llms.txt | grep -o '<title>[^<]*</title>'
       curl -fsS https://affine.io/public/read_links.json | python3 -c 'import json,sys,time; d=json.load(sys.stdin); print(d["head_version_id"], d["expires_at"]-int(time.time()), "seconds left")'
       ```

       - Every line of the loops starts with `200`. The pages are `text/html`, the `.json` files are `application/json`, `llms.txt` is `text/plain`.
       - The second loop prints five lines: two fonts, `favicon.svg`, `site.css`, `site.js`.
       - The last command prints the head and about 86000 seconds. Take one small file from `base.files` of `public/read_links.json`, download it with its `url`, and compare the sha256 and the size.
       - In a browser, `https://affine.io` shows the v2 page with the 10 kings and the queue, and `https://affine.io/v1.html` shows v1's history.
       - If a file is on the disk under `$SITE/public/` and its address gives 404, the real web server does not serve the new directory. Stop and tell the coordinator. Go back with "Go back" if the root page is broken.
       - Look at the `cache-control` and `age` headers of `https://affine.io/public/read_links.json` (`curl -sI`). If a cache holds the file for more than a few minutes, tell the coordinator: a links file that is cached for longer than its life gives dead links.
    2. **The unit and the timer: one run each minute.** From here the site follows the store.

       ```
       for unit in sn120-v2-site-pull.service sn120-v2-site-pull.timer; do
         sed -e "s|SN120_USER|$RUNUSER|g" -e "s|SN120_ENV_FILE|$PENV|g" -e "s|SN120_PYTHON|$PY|g" \
             -e "s|SN120_V2_DIR|$REPO/v2|g" -e "s|SN120_SITE_DIR|$SITE|g" -e "s|SN120_PUBLIC_URL|https://affine.io|g" \
             -e "s|SN120_V1_HISTORY|$V1FILE|g" "$REPO/v2/systemd/$unit" | sudo tee "/etc/systemd/system/$unit" > /dev/null
       done
       systemd-analyze verify /etc/systemd/system/sn120-v2-site-pull.service /etc/systemd/system/sn120-v2-site-pull.timer
       sudo systemctl daemon-reload
       sudo systemctl enable --now sn120-v2-site-pull.timer
       ```

       Check: `systemctl list-timers sn120-v2-site-pull.timer --no-pager` lists the timer. After two minutes, `journalctl -u sn120-v2-site-pull.service -n 5 --no-pager` shows `failed steps 0`.

       **Not landed yet:** the unit loads `$PENV` as it is, and the pull of this commit reads only the `SN120_R2_` names. With a build that lacks the change, the log shows an error about a missing `SN120_R2_ENDPOINT` and no run writes anything: disable the timer (`sudo systemctl disable --now sn120-v2-site-pull.timer`), tell the coordinator, and do not edit the credentials file. The site stays as the run by hand left it.
    3. **Move v1's stale files out of the site directory.** After the switch, v1's data files still answer at their old addresses and look live. This moves them to `$MOVED`. Nothing is deleted.

       ```
       "$PY" "$REPO/v2/site/reset.py" out --site-dir "$SITE" --to "$MOVED" --scores --dry-run
       "$PY" "$REPO/v2/site/reset.py" out --site-dir "$SITE" --to "$MOVED" --scores; echo "exit=$?"
       ```

       - `--to` is the directory that takes the files. It must be outside the site directory and on the same disk.
       - `--dry-run` prints what would move and changes nothing.
       - `--scores` also moves `scores.json`. v1's weight writer reads that file. Give `--scores` only when v1's weight writer is stopped (step 9). Without the option, `scores.json` stays.
       - The tool refuses when the v2 site is not in the directory yet. So it cannot leave the root with no page.
       - It keeps a list of what stays, and moves everything else.
       - What stays: the v2 site's files, `llms.txt`, `mining.json`, `public/`, `ff/links/`, `ff/links.json`, `llms-page.html`, `network.css`, `robots.txt`, two images, and v1's `.js` and `.css` files.

       Check:

       - Read the dry run first. No line names `ff/links`, `public`, `llms.txt`, `mining.json`, `llms-page.html` or `network.css`.
       - The real run prints `reset out: moved N of N entries to ...` and `exit=0`. A line `reset: refused: ...` and exit code 2: nothing changed.
       - `for p in /k1.json /scores.json /benchmarks.json /ff/status.json; do curl -s -o /dev/null -w "%{http_code} $p\n" "https://affine.io$p"; done` prints `404` four times.
       - Run the two loops of 12.1 again: every line still starts with `200`.
       - The next run of the pull still shows `failed steps 0`.

    Later, only when the way back is no longer needed, move v1's scripts and styles out too:

    ```
    "$PY" "$REPO/v2/site/reset.py" out --site-dir "$SITE" --to "$KEEP/moved-assets" --scores --assets; echo "exit=$?"
    ```

    They stay until then because the edge can keep an answer for a `.js` or `.css` address with `?v=` for one day, also a "not found" answer. After a way back, v1's page would then load a kept "not found" instead of its script. Check: `exit=0`, and the title line of 12.1 still shows a page title.
13. **Read the public files as a miner does.**

    ```
    curl -fsS https://affine.io/llms.txt | grep 'Where to get the link'
    ```

    Check: the line names `https://affine.io/ff/links/<your hotkey>.json`. If it says `not published yet`, the GPU box runs without `SN120_LINKS_URL`: set it there, start `run` again, and do not announce before this check passes.

## The chain check

A script for step 6 and step 9 of "AffineValidator". It only reads the chain. It needs no wallet. Paste the block as it is, from the first column:

```
cat > "$HOME/sn120-v2-chain-check.py" <<'EOF'
import bittensor
from bittensor import storage, timelock

NETUID, VALIDATOR = 120, "5HmYnmUYT6qe3yFMg1Ad8WLLqDvnwtjYakXBpDvoRW1Qqzb8"
item, chain = storage.SubtensorModule, bittensor.Subtensor("finney")
block = int(chain.block_info(chain.block).number)


def read(name, *keys):
    return chain.query(getattr(item, name), [NETUID, *keys], block=block)


uid = next(int(u) for u, hotkey in chain.query_map(item.Keys, [NETUID], block=block) if str(hotkey) == VALIDATOR)
last, limit, tempo = int(read("LastUpdate")[uid]), int(read("WeightsSetRateLimit")), int(read("Tempo"))
epoch, epoch_start = int(read("SubnetEpochIndex")), int(read("LastEpochBlock"))
print("block:", block, "| validator UID:", uid)
print("commit-reveal on:", bool(read("CommitRevealWeightsEnabled")), "| reveal period in epochs:", int(read("RevealPeriodEpochs")))
print("rate limit in blocks:", limit, "| last write block:", last, "| next write allowed in blocks:", max(0, last + limit - block))
print("next epoch in blocks:", max(0, epoch_start + tempo - block))
for index in (epoch - 1, epoch):
    for commit in read("TimelockedWeightCommits", index) or []:
        if str(commit[0]) == VALIDATOR:
            print("commit that waits for its reveal: block", int(commit[1]), "| opens at", timelock.reveal_time(int(commit[3])))
row = sorted((int(u), int(w)) for u, w in read("Weights", uid))
total = sum(w for _, w in row) or 1
print("weights on chain:", {u: round(w / total, 4) for u, w in row})
EOF
```

Run it with the virtual environment of the checkout: `"$REPO/v2/.venv/bin/python" "$HOME/sn120-v2-chain-check.py"`.

| Line of its output | Meaning |
| --- | --- |
| `commit-reveal on`, `reveal period in epochs` | The weights go to the chain hidden, and the chain opens them after that many epochs. An epoch is 360 blocks, 72 minutes. |
| `rate limit in blocks`, `last write block`, `next write allowed in blocks` | The last weight write of the validator hotkey, by any writer, and how long the chain refuses the next one. |
| `commit that waits for its reveal` | A write that the chain has accepted and not opened yet. |
| `weights on chain` | The weights that count now. |

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

On AffineValidator: `journalctl -u sn120-v2-weights.service --since "3 hours ago" --no-pager` shows the last runs. The dry run of step 5 of "AffineValidator" shows `blocks_since_last_write`; one hour is 300 blocks, so it stays below about 310 while each hourly run sets. If the first log line of a run says `read from the cache of the journal copy`, the store did not answer or the store settings are wrong: the writer then pays the last kings that it read.

### After a fix: a new commit on the GPU box

A round that an older commit opened continues at a newer commit. A move in the middle of a round costs only that the round runs again from its start.

One case needs one more command: the fix changed the code digest or the install, and a round is open. Then end the round first. Without `end-round`, the jobs of the old round are refused 3 times each, and every seat gets one counted failure.

1. Stop `run`.
2. Check out the commit, install again and run the test suite (checks 1 and 2 of "Before the mainnet bootstrap").
3. If a round is open and the digests changed, end the round:

   ```
   python -m sn120_v2 end-round --state-dir $STATE --stack sn120_v2.assembly:stack
   ```

   It ends an open, undecided round at once. Nobody is charged, no failure is counted for a seat, and the seats are first in the next round. It is refused (exit code 2, `no round is open and undecided`) when no round is open or the round is already decided; `run` finishes a decided round.
4. The digest block (check 3 of "Before the mainnet bootstrap").
5. `set-params` with a file that holds the new digests. The file holds the whole set of overrides.
6. Start `run` again. Never run `bootstrap` again.

`python -m sn120_v2.gpu.plan` only prints: run it before step 3 and compare `code_digest` and `image_digest` with `$CUT/params.json` to see if the digests changed. `status` shows a `phase` other than `IDLE` while a round is open.

After any failed lease, the next round opens 5 minutes later at the earliest (`apply_retry_s` 300).

### A new GPU box

1. Stop `run` on the old box, if the box still answers.
2. On the new box: the checkout, the install, step 2 of "GPU box" with the same store settings and a new empty state directory, and `SN120_IMAGE_DIGEST` from `python -m sn120_v2.gpu.plan`.
3. If the new box gives other digests than the pinned parameters hold, pin them first with `set-params`.
4. Start `run`. It pulls the journal copy and continues. Do not run `bootstrap` again.

Two `run` processes never both write: the newer one takes over the journal, and the older one stops with `FencedError` at its next write. Do not let a supervisor start the old one again.

### Go back

Before the first v2 round ends. The weights part is "To go back" of the handover: stop v2's timer first, then start v1's writer.

1. **Stop v2's writers.** Stop `run` on the GPU box. On AffineValidator:

   ```
   sudo systemctl disable --now sn120-v2-weights.timer sn120-v2-site-pull.timer
   systemctl is-active sn120-v2-site-pull.service
   ```

   Check: the second line prints `inactive` or `failed`. If it prints `activating`, wait and ask again. A pull that still runs replaces the files again.
2. **Only if v1's `affine-ff.service` comes back: stop the v2 links job.** Two link jobs never write to the same place.

   ```
   sudo systemctl disable --now sn120-v2-links.timer
   systemctl is-active sn120-v2-links.service
   ```

   Check: `inactive` or `failed`.
3. **Put v1's files back from the copy.**

   ```
   "$PY" "$REPO/v2/site/reset.py" back --site-dir "$SITE" --from "$BACKUP" --dry-run | tail -1
   "$PY" "$REPO/v2/site/reset.py" back --site-dir "$SITE" --from "$BACKUP"; echo "exit=$?"
   ```

   - `--from` is the full copy of step 4 of "AffineValidator". `$MOVED` is not needed: the copy is complete.
   - Each file is copied under a temporary name and then renamed. `index.html` goes last, after the files that it loads.
   - It copies only a file that is missing or different. It never puts an old link file back. It refuses a `--from` directory that holds the v2 page.

   Check: `reset back: copied N files from ..., 0 failed` and `exit=0`.
4. **Move the pull's directory away**, if step 4 of "AffineValidator" showed that v1 had no `public/`.

   ```
   mv "$SITE/public" "$KEEP/public-v2-$(date -u +%Y%m%dT%H%M%SZ)"
   ```

   Check: `curl -s -o /dev/null -w '%{http_code}\n' https://affine.io/public/mining.json` prints `404`.
5. **Start v1's units again**: v1's weight writer, and the units that steps 8 and 10 stopped, with `systemctl --user enable --now <unit>`. Their files in the site directory are as old as the copy; each unit makes its file fresh at its next run. `affine-ff.service` writes the link files again at its next run.

   Check: `curl -fsS https://affine.io/k1.json | "$PY" -c 'import json,sys,time; print(int(time.time()-json.load(sys.stdin)["generated_at"]), "seconds old")'` prints a small number after `affine-k1.service` ran.
6. **Check the site.**

   ```
   curl -fsS https://affine.io/ | grep -c 'site.js?v='
   curl -fsS https://affine.io/llms.txt | head -3
   curl -fsS https://affine.io/llms.txt | sha256sum; sha256sum "$BACKUP/llms.txt"
   ```

   Check: `grep -c` prints `0`. The two hashes are equal. In a browser, `https://affine.io` shows v1's page with its charts.

Five v2 files stay in the site directory after the way back: `site.js`, `site.css`, `v1.html`, `site-data.json`, `v1-data.json`. No v1 page links them. Remove them with `rm` when you want, or leave them for the next switch.

Forward again after a way back: do steps 10 and 12 again. The copy and the frozen history of step 4 stay valid. Give `reset.py out` a new `--to` directory: the tool refuses a directory that already holds one of the names.

The import only read public files, and v2 writes no key that v1 wrote, so v1's state in the bucket did not change. The objects that v2 wrote stay in the bucket under v2's own keys. The weights that are on chain stay until the next write.

Never, on the site:

- Never run the pull, by hand or by the timer, before step 4 and before the import's real run.
- Never delete or rename the site directory itself. v1's web server mounts it at its start, and the links job needs it.
- Never move `llms-page.html` or `network.css`: the web server makes the browser view of `/llms.txt` from them.
- Never move `scores.json` while v1's weight writer runs.
- Never copy an old link file back by hand. The links job writes the full set on every run.
- No DNS change, no caddy change and no restart of the web server. None is needed.

## Upload links job

A miner uploads its file through a link that only its hotkey can open. The job writes one sealed link file for each waiting hotkey that made the one-time activation on chain, and an index. It reads the queue from the journal copy in the store. One run takes about 30 seconds; the timer starts a run at minute 10 and minute 40 of each hour, and a link works for 24 hours.

- **`--credentials-file`** names v1's credentials file. The job reads the two parts of the token from it on each run (`R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`, v1's names). Nothing copies the secret, and systemd never loads the file. The token must be able to write: a link signed with a read-only token is refused at the upload. The bucket and the endpoint are `Environment=` lines of the unit; the unit has no prefix line.
- **`--state-dir`** is a directory of the job's own. It is not the directory of the weights writer or of the pull.
- **`--site-dir`** writes `ff/links/<hotkey>.json` and `ff/links.json` under the site directory. These are the paths and the format of v1, so the address of a miner's link file does not change. The job removes each other `.json` file in `ff/links/`: the link file of a hotkey that has no seat in v2.
- **`--network finney`** reads the activations. When the chain cannot be read, the run fails and writes nothing; the old links work for up to 23 hours more.

A failed run prints `issue failed: <reason>`, writes nothing and removes nothing.

**Do not replace the token in v1's credentials file without one test upload from a machine outside the host.** v1's notes record a token with a limit on the source address: its links worked on the host and gave HTTP 403 to every miner. The token that is in the file now works for the miners. After a replacement, every link signed with the old token stops at once: start one run (`sudo systemctl start sn120-v2-links.service`), then make the test upload.

**Why `affine-ff.service` must be stopped before the first run.** The v1 service and this job write the same files in the same directory. Every 30 minutes v1 writes `ff/links/<hotkey>.json` and `ff/links.json`, and removes the file of each hotkey that is not eligible in v1's view. v1's view of the queue is frozen since the teardown. There is no lock between the two. If both run:

1. v1 removes the link files of hotkeys that only v2 knows, and v2 removes those of hotkeys with no seat in v2. A miner gets HTTP 404 at random.
2. The index `ff/links.json` changes between two lists.
3. v1 writes `ff/status.json` again after the runbook moved it away.

A link that v1 issued before the stop works until its own end, 24 hours at most. It points to the same object that v2 reads (the same bucket, no prefix, the same key), so a file uploaded through it reaches v2. The activations are on chain: nothing is copied.

Never run two link jobs that write to the same place. Never run `links activate --execute` as the operator: it is the miner's own chain write. Never paste an opened link into a log or a chat. Never copy the credentials file, and never show its content.

## Public pull and site

Nothing serves the bucket to the public. The pull (`site/pull.py`) is the one way from the store to `https://affine.io`. One run, then exit; the timer starts a run each minute, at second 30.

What one run does:

1. It copies `public/mining.json` and `public/llms.txt`, only as a pair of the same snapshot. Each goes to the path of its key and also to the root of the site, so `https://affine.io/llms.txt` stays the address of `llms.txt`.
2. It copies the head pointer and writes `public/read_links.json`. The links are for the base, the LoRAs and the master of the current head, and each works for 24 hours (`--ttl-s`).
3. It copies the head manifests, the round files, the task tables that `mining.json` names, and `public/benchmarks.json` when it exists.
4. It builds the status site (`site/build.py`): the v2 page, and `v1.html` from the frozen v1 history.

What it may touch in the site directory:

- Under `public/`, only files that its own list `public/pull_index.json` names. A file that it did not write is left as it is, and the run exits 1 with the name of the file.
- At the root, the eleven files that step 12 of "AffineValidator" names. It replaces them with no check of who wrote them. So the first run after the bootstrap is the site switch.
- Nothing else: not `ff/links/`, `ff/links.json`, `k1.json`, `scores.json`, `k1/`, or any other file.

How it knows v2's keys: it reads four fixed keys (the constants of `sn120_v2/layout.py`), lists only `public/heads/` and `public/rounds/`, and takes from a list only a key that a function of `layout.py` builds. The head pointer must be a v2 head manifest; if v1's object is at that key, the run fails at that step and writes no pointer and no links file. This is one more reason for check 4 of "Before the mainnet bootstrap".

Disk: the pull keeps the newest 200 rounds (`--keep-rounds`) and removes its own older round files. The worst case is 200 times 64 MiB; the real size is far less.

Stop: `sudo systemctl disable --now sn120-v2-site-pull.timer`. The files stay as they are.

## v1 units

The names come from documents in this repository. The repository does not show which of them run now. The host is the truth: `systemctl --user list-units 'affine-*' 'sn120-*' '*watchdog*'` and `systemctl --user list-timers --all`.

**Group A: the round path. Torn down on 13 Nov 2026; check that each is inactive and disabled.**

1. The watchdogs: `tp-watchdog`, `ff-watchdog`, and any other unit that `systemctl --user list-units '*watchdog*'` shows. A watchdog reverts v1 or starts it again when the lifecycle is down for 90 seconds (`ops/free_file/README.md`, `ops/ff_train_params/README.md`).
2. `affine-mimo-lifecycle.service`: selects the next turn and starts the training.
3. `affine-promotion-gate.service`: judges a turn. With it stopped, the set of turns with no verdict does not change.
4. `affine-mimo-trainer-supervised.service` and `affine-mimo-evaluator-supervised.service`, with their `-tunnel` units (`ops/pods/roles.json`).

**Group B: the weights writer ("AffineValidator", step 9).**

- The repository names the hourly writer in two ways. `docs/MIMO_DEPLOYMENT_RUNBOOK.md` names `affine-live-reward-writer.timer` and `affine-live-reward-writer.service`. The K1 service (`ops/k1_pass_rates/services/k1_service.py`) says that the hourly writer is `weights_flip.py`, which reads `website/scores.json`; no document gives the unit name of that script.
- So find the writer on the host: `systemctl --user list-timers --all`, and `grep -rlE 'weights_flip|live_reward_writer|current_assessment_writer|set_weights' ~/.config/systemd/user/`. Stop and disable each timer and service that these show.
- `affine-hourly-burn.timer` and `affine-transition-weights.timer` are old writers (`ops/HOURLY_BURN.md`, `ops/EQUAL_REGISTRATION_WEIGHTS.md`). They must stay disabled and masked.

**Group C: upload links, publishers and side services ("AffineValidator", steps 8 and 10).** Never run two writers of one public file.

- `affine-network-dashboard.service`: writes `network-data.json` every 15 seconds, and writes v1's `llms.txt` and `reward-policy.json` back when they differ from its own copies. It stops first, in step 8, before the new `llms.txt` goes up.
- `affine-ff.service`: issues the miners' upload links, checks their files, writes `ff/links/`, `ff/links.json` and `ff/status.json`.
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
