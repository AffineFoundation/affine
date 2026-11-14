# Cutover: start v2 on mainnet from v1's state

This is the operator's runbook for the mainnet cutover. Do the smoke test first: [SMOKE.md](SMOKE.md). Cut over only after it passes.

## Who does what

- **Jacob (the operator)** does every step on a host, a wallet or the chain. No step is automatic.
- **The coordinator** archives the v1 state of `main` as the branch `v1-archive` and the tag `v1-final`. Then `v2` takes over `main`. Only the coordinator does this. The coordinator names the commit for the cutover.
- **No worker** touches a host, a wallet, the chain or `main`.

## Not in this file yet

Each point is a fact or a command that is not on `v2` at the commit of this file. Do not guess it. Each place below that needs one says **Not landed yet**.

1. The numbers from the smoke test for `params.json`: `recipe.global_norm_cap` (it waits for the update norm that the smoke test measures), and any size that the smoke test changes.

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

## The state of v1 (14 Nov 2026)

**v1 still takes turns, and it goes on until its units are stopped.** v1's evaluation was reported as torn down on 13 Nov 2026 at 4:55 PM Costa Rica time (22:55 UTC). The public `k1.json` does not show that: v1's lifecycle still makes an epoch about every 15 minutes. It starts a new turn when a seat has a valid file, and with no valid file it writes a skip. On 14 Nov 2026: epoch 1320 a skip at 00:34:41 UTC, epoch 1321 a turn for UID 893 at 00:38:49 (verdict `pending`), epoch 1322 a skip at 00:54:11, epoch 1323 a skip at 01:09:29. So the part "stop v1's rounds" is not done. What also runs: v1's hourly weight writer, `affine-ff.service` (upload links) and the publishers of the v1 site. Jacob stops each of them at its step of the list "AffineValidator".

**The sign that anybody can read:** a new row in `turns` of `https://affine.io/k1.json` about every 15 minutes means that v1's lifecycle runs. The sign works only while `affine-k1.service` runs: that unit writes `k1.json` (point 2 below). This line prints the last four rows (epoch, `turn` or `skip`, UID, time in UTC):

```
curl -fsS https://affine.io/k1.json | python3 -c 'import json,sys,time
for r in json.load(sys.stdin)["turns"][-4:]: print(r["epoch_index"], r["status"], r.get("uid"), time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(r["turn_at"])))'
```

On 14 Nov 2026 at 01:11 UTC its last line was `1323 skip None 2026-11-14 01:09:29`.

What the operator checks, on the host where v1's round path runs (`systemctl --user`). The order is the same as before: first the units, then the last epoch, then the import.

1. No unit of group A in "v1 units" is active or enabled. While one is, v1 goes on. The watchdogs come first: a watchdog that still runs starts v1's round path again. By v1's code in this repository, each watchdog has ended by itself, so the result to expect is a list with no watchdog that reverts v1 ("v1 units", group A, point 1).
2. v1's last epoch does not move. Run the line above two times, 20 minutes apart, so that the time of one v1 epoch lies between them: the last line must be the same. Then run the import dry run ("GPU box", step 4) two times in the same way: `v1_last_epoch` in the `source:` line, the kings and the front seats must be the same. If the last epoch moved, the lifecycle still runs: go back to point 1. Not proven: that the last epoch stands still when no unit of group A runs. If it still moves then, send the output to the coordinator.

   **`k1.json` is a proof only while `affine-k1.service` runs.** That unit writes `k1.json`. Once it is stopped (step 10 of "AffineValidator"), `k1.json` stands still, whatever the lifecycle does. Then the proof that v1 takes no turn is the state of the unit: `systemctl --user is-active affine-mimo-lifecycle.service` prints `inactive`, and `systemctl --user is-enabled affine-mimo-lifecycle.service` prints `disabled`.

**The numbers in this file are those of the moment.** They move while v1 runs and while miners register. On 13 Nov 2026 at 23:04 UTC the dry run gave `v1_last_epoch=1314`, 10 kings, 14 front seats and 256 waiting seats. On 14 Nov 2026 at 01:00 UTC, v1's files of epoch 1310 and the chain of that minute gave 257 waiting seats (14 front, 49 normal, 194 back), and 8 minutes later 258. Check a number against the output of your own run, not against this file.

**Save `k1.json` and `mining.json` now**, and again right before the import's real run. `k1.json` shows only v1's last 200 epochs, and it is the only record of a turn that v1 took and did not judge. The import needs the two files from the same time ("GPU box", step 7), so save them together. Keep all copies.

```
T=$(date -u +%Y%m%dT%H%M%SZ)
curl -fsS https://affine.io/k1.json -o k1-$T.json
curl -fsS https://affine.io/mining.json -o mining-$T.json
python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("epoch"))' mining-$T.json
```

Check: the last line prints v1's epoch name, which ends with `-e` and a number (for example `-e001311`). If it prints `None`, the site serves v2's `mining.json` and this copy is not a v1 file: keep the last copy that is.

The first copies are in this repo: `tests/data/cutover/k1.json` and `tests/data/cutover/mining.json`. This `k1.json` was made on 13 Nov 2026 at 22:13 UTC, and its last epoch is 1310.

**The import dry run can run at any time, as long as the site serves v1's `mining.json`.** It reads only public files and the chain. Do the real run on the cutover build, right before the bootstrap. Once the site serves v2's `mining.json`, the import with the two public URLs is refused: "GPU box", step 7 has the way.

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

1. Check out `0c29cb6c`, or a later commit that the coordinator names, and install again. The packages can differ.

   ```
   git fetch origin && git checkout 0c29cb6cb59f5b66c44b482cfe77b5127b673d1f
   git rev-parse HEAD        # write this down with the cutover records
   for commit in 505f5ddab15c325ecedcf5c763bddf61225ff6ff 9b830458d90c703497558fb1d401938b097a157a 5ce10ce7cc3ae25d87926d237f9a7e215999aea2 97a67ea4f80a241418fafb905222caa61354e55a 2b9ee5d1615b994a89a0072a8a12b3016be07ac8 4e23a27526e3eb700e274d1a3eea3f93e84e8277 e0faff2dfbcff716723d39deb7e0670e7bd4e7b0 f986ccb11c9627cf2109d71c5480f827fde78acd 0c29cb6cb59f5b66c44b482cfe77b5127b673d1f; do
     git merge-base --is-ancestor $commit HEAD; echo "$commit exit=$?"
   done
   cd v2 && source .venv/bin/activate
   uv pip install -e ".[dev,gpu,r2,chain,envs,miner]"
   ```

   **The mainnet bootstrap and `run` use only a commit that contains these nine.** Each line of the loop must end with `exit=0`.

   | Commit | What it holds | With an earlier commit |
   | --- | --- | --- |
   | `505f5dda` (PR #131) | The head pointer at `public/head.json`, and the preflight command. | The build writes the head pointer to `public/current.json`, which is v1's object. |
   | `9b830458` (PR #145) | The time limit of a round as a hard stop, and the verdict `not_scored` ("The time limit of a round" below). | A round can run past its time limit. `llms.txt` must not go up from such a build. |
   | `5ce10ce7` (PR #146) | The limit of 3 checks for a file that makes the checker fail, with 10 minutes between two checks of the same file. | A hostile file is checked again at every pass. `llms.txt` must not go up from such a build. |
   | `97a67ea4` (PR #149 and PR #151) | The restart loop of `run` ("GPU box", step 10). | Nothing starts `run` again after a crash or a reboot. |
   | `2b9ee5d1` (PR #154) | The loop waits when the store or the chain does not answer, without end. | The loop gives up after 5 such starts, about 2.5 minutes. Then nothing starts `run` again until you come. |
   | `4e23a275` (PR #155) | The text of `llms.txt`, corrected after a check of the whole text against the code. For example: a hotkey gets one turn, and a new entry needs a new hotkey. | `llms.txt` tells a miner things that the code does not do. `llms.txt` must not go up from such a build. |
   | `e0faff2d` (PR #157 and PR #159) | The guard of `run`: it compares the box with the pinned digests before it opens a round, before it runs an open round again and before a fold. On a difference it starts none of them and ends with exit code 2 ("The digests of the box and the pinned digests" below). | `run` does not stop on such a box. Each round closes in seconds, and every seat in it gets one counted failure. |
   | `f986ccb1` (PR #168; it holds PR #164, `6baf0d74`, PR #166, `68b8abc5`, PR #163, `d32cd947`, PR #161, `c87b81ea`, and PR #160, `e9958463`) | Six things for the operator. From `e9958463`: a chain read of `run` ends in 90 s at most, and after a failed read the next pass opens a new connection. From `c87b81ea`: the command `release-seat`, and the keys `held_seats` and `seats_with_counted_failures` of `status` ("After a fault of the box in training" below). From `d32cd947`: the command `force-sync` ("When `run` refuses the chain sync" below). From `68b8abc5`: a round in which no training finished counts no failure for a seat ("A round in which no training finished" below). From `6baf0d74`: `set-params` prints `changed`, each parameter whose value differs, with the value before and after ("The value `squat_min_wait_s`" below). From `f986ccb1`: a text from an error is made safe before it goes into the public round file ("Error text in the public round file" below). | A chain read that gets no answer ends only when the chain library connects again by itself (after 303 s in the slowest case that was tried). No command releases a held seat. No command applies a chain sync that `run` refuses, so no round opens after such a refusal. A box on which every training fails gives every seat of each round one counted failure, and 3 hold a seat. `set-params` prints only the two hashes, so nothing shows a key that went back to its default. A text from an error goes into the public round file as it is: it can hold an address of the box or the signature of a signed link. |
   | `0c29cb6c` (PR #171) | The words of `llms.txt` on `no_decision`: a failure is counted for a seat in one case only (the box could not train its batch, and another batch of the same round finished its training); a round with no result costs a seat nothing; the markers in a public `reason` or `note`. | `llms.txt` tells a miner that each `no_decision` of the box is counted and that 3 of them hold the seat. The code counts fewer cases. `llms.txt` must not go up from such a build. |

   With `exit=1` on a line, the checkout is an earlier commit. Stop, and do not run `bootstrap`, `run` or `publish` with the production store settings. Any other number means that git does not have that commit: run `git fetch origin v2` and try again. Check 4 below reads the first fact from the code.

   **`0c29cb6c` is the least commit for mainnet.** It is the last of the nine and holds the other eight. The coordinator named it as the commit to take before the first mainnet round. The reasons are the last two rows: the safe error text in the public round file (`f986ccb1`) and the words of `llms.txt` on `no_decision` (`0c29cb6c`). Take a later commit only when the coordinator names it. A later commit that changes only `v2/CUTOVER.md` changes nothing on a box: read this file at the tip of `v2`.

   Each change on `v2` from `1d7d0628` to `0c29cb6c` is a checkout only: no install line, no digest block, no `set-params`, no new bootstrap. No file under `sn120_v2/gpu/` changed, and `pyproject.toml` did not change. So for a box that is at `1d7d0628` or later: stop the loop, check out, run the `diff` line of "The digests of the box and the pinned digests", and start the loop. `run` loads the new code only at a start. The steps are in "A checkout only: a running box goes to a later commit" in "After the start".

   **There is no way back below `9b830458`.** After one round with a `not_scored` entry, no older commit can read the state. Never check out an older commit on a box or a host that reads this state, not even for a test.

   If the box still holds the state of the smoke test at an older commit, and the smoke test goes on: a move to a commit with `9b830458` or `5ce10ce7` is a checkout only. No install line, no digest block, no `set-params`, no new bootstrap. Stop `run` between two rounds, when `status` shows `"phase": "IDLE"`; check out; start `run` again. For a later commit this is not sure. After the checkout and before the start, run `python -m sn120_v2.gpu.plan | diff - $SMOKE/plan.json`. An output means that the digests of the box changed: then the digest block, `smoke params` and `set-params` of SMOKE.md come before the start ("The digests of the box and the pinned digests" says why).

2. The test suite and the CPU check. Pass: `pytest` shows no failure; `sim` prints `"replay_matches": true`. Run them in a new shell that has no `SN120_` variable of the smoke test (`env | grep ^SN120_` prints nothing).

   ```
   python -m pytest
   python -m sn120_v2 sim --state-dir /tmp/sn120-sim-cutover --rounds 6 --miners 4
   ```

3. The digests. A commit that changes a file under `sn120_v2/gpu/` has a new code digest, and new packages give a new image digest. Run this block for the cutover build in any case: with stale digests of the smoke test, `run` is refused before the first round ("The digests of the box and the pinned digests" below).

   ```
   export CUT=<a local disk with 300 GB free>/sn120-mainnet
   mkdir -p $CUT
   python -m sn120_v2.gpu.plan | tee $CUT/plan.json
   export SN120_IMAGE_DIGEST=$(python -c "import json,os; print(json.load(open(os.environ['CUT']+'/plan.json'))['image_digest'])")
   nvidia-smi --query-gpu=index,name,memory.used --format=csv     # 8 GPUs, almost no memory in use
   ```

4. Three changes that the build must hold. Stop if one of them is wrong.

   ```
   python -c "from sn120_v2 import layout; print(layout.CURRENT)"
   grep -c SN120_LINKS_URL sn120_v2/assembly.py
   python -m sn120_v2.r2 --help | grep -c preflight
   ```

   - The first line must print `public/head.json`. With `public/current.json`, this build would overwrite a v1 object.
   - The second must print a count of 1 or more. With 0, this build does not read `SN120_LINKS_URL`, and `llms.txt` does not name the link files.
   - The third must print a count of 1 or more. With 0, this build has no preflight command.

5. A new parameter file, `$CUT/params.json`. Never give the mainnet bootstrap a file of the smoke test. The six small files of `smoke params` (`params-1-tiny.json` to `params-4-multi.json`, `params-limit-judged.json`, `params-limit-not-scored.json`) set values that are for the smoke test only: a win rule of `z_bar` 0, a few tasks, short time limits. `params-5-full.json` and the file of step 4 of SMOKE.md hold the digests of the smoke commit. The coordinator names `global_norm_cap` (about 4 times the update norm that the smoke test measured, SMOKE.md step 9) and any other value that the smoke test changed.

   ```
   {"squat_min_wait_s": 31536000, "recipe": {"image_digests": ["<image_digest of plan.json>"], "code_digest": "<code_digest of plan.json>", "global_norm_cap": <number>}}
   ```

   `"squat_min_wait_s": 31536000` (365 days, in seconds) switches squat-trim off for the seats that come from v1. Squat-trim is the rule that moves a waiting seat with no valid file to the back of the queue.

   - With the default (86400, one day), a seat whose registration is older than 2 days and that has no valid file in v2 moves to the back of the queue, and stays there. This happens at the first try to open a round after the first charged entry. A seat from v1 has no valid file in v2 when it has only v1 `train` keys, or no file. On 14 Nov 2026 (chain block 9501013), 963 of the 1,000 registrations on the chain were older than 2 days.
   - Jacob's rule (13 Nov 2026, 4:33 PM): such a hotkey keeps its place, not charged, until the miner uploads a new file.
   - With 31536000, no seat moves whose registration is younger than 730 days. On that day the oldest registration was 54 days old. (UID 0, the owner hotkey, is older, but it has no seat.)
   - Only two things read the value: the step that opens a round, and the text of `llms.txt`. No range is checked for it.
   - The key stands at the top level, not inside `recipe`, and the number has no decimal point. A file with one of these faults is refused.
   - Jacob decides on this value. Without the key, the default holds.

   Check the file before the bootstrap:

   ```
   python -c "import os; from sn120_v2.params import load_params; p = load_params(os.environ['CUT'] + '/params.json'); print(p.squat_min_wait_s, p.params_hash)"
   ```

   It prints the value and the hash of the whole file's parameters. The value must be `31536000`. `86400` means that the key is missing. An error that ends with `unknown field(s) ['squat_min_wait_s'] for Recipe` means that the key is inside `recipe`. Write the hash down: after the bootstrap, `"params_hash"` of `status` must be the same ("GPU box", step 8).

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
   - The 10 kings are the miners that v1 pays now. The report shows them two times: under `kings, oldest first`, and in `first weights` in the order of the UID. Compare the two as a set.
   - The front seats are the turns that v1 did not judge. Each line shows the v1 epoch of the turn, and how many turns that hotkey had in v1. The seat of epoch 1157 (UID 923) must be among them.
   - `first weights` must show 10 UIDs.
   - The line `UID 0 is <address>: it has no seat and gets no weight; it is the given owner hotkey` must be there. It is the last line of the report. After it comes only `DRY RUN: nothing was written.` (in the real run of step 7: the five `WROTE` lines).
   - A refusal with `mining.json and k1.json are not from the same epoch`: the site serves v2's `mining.json`. Step 7 has the way.
5. **Preflight (hard gate).** A read-only command lists every key and prefix that v2 writes, and fails when one of them already exists in the bucket. Run it before the first production write, which is step 6. If it fails, stop the cutover: v2 would write over an object that is not its own. Do not go on without it.

   Nobody has run it against the real bucket: this run is the first. It only lists; it writes nothing.

   ```
   python -m sn120_v2.r2 --preflight; echo "exit=$?"
   ```

   The output on a bucket and prefix with nothing of v2, and `N` miner files of v1:

   ```
   preflight  bucket 'affine-verification-mock-20260930', key prefix '': 16 keys and prefixes from layout.py, list calls only
   ok      pending  nothing
   ok      private/ff/v1/miners  N objects: the miner files, read in place
   ok      private/journal  nothing
   ok      private/rounds  nothing
   ok      public/bases  nothing
   ok      public/benchmarks.json  nothing
   ok      public/ff/links  nothing
   ok      public/ff/links.json  nothing
   ok      public/head.json  nothing
   ok      public/heads  nothing
   ok      public/irt  nothing
   ok      public/llms.txt  nothing
   ok      public/loras  nothing
   ok      public/masters  nothing
   ok      public/mining.json  nothing
   ok      public/rounds  nothing
   PASS  no object at a key that v2 writes
   exit=0
   ```

   Check:

   - `exit=0`, and the last line of the command is `PASS  no object at a key that v2 writes`.
   - The first line names the production bucket and the key prefix `''`.
   - The line for the head pointer says `public/head.json`. If it says `public/current.json`, the commit is too old (check 1 of "Before the mainnet bootstrap").
   - The line for `private/ff/v1/miners` shows the count of miner files. v2 reads them in place and never writes there, so objects there are expected.

   When it does not pass:

   - A line `EXISTS  <key or prefix>  N objects`, then up to 10 of the keys, and the last line `FAIL  objects exist at keys that v2 writes: do not start v2 on this bucket and prefix`, with `exit=1`. Stop, and send the output to the coordinator. Do not delete anything to make it pass.
   - A line `FAIL    <key or prefix>  <error>`: that list call failed, so the place is not known to be empty. A line `FAIL  open the bucket  <error>`: the store settings of step 2 are wrong. Correct the cause and run it again.

   What the command does: it sends only list calls (`ListObjectsV2`), so it changes nothing, and it reads only the `SN120_R2_*` names of step 2. It has run against a local stand-in for the store only.

   **An early run, from the shell of the smoke test, before the cutover day.** The smoke test uses the same bucket under its own key prefix. This line looks at the root of the bucket for one command; the shell keeps the prefix of the smoke test:

   ```
   SN120_R2_PREFIX= python -m sn120_v2.r2 --preflight; echo "exit $?"
   ```

   It needs a commit with `505f5dda`. A pass is the output above: the first line names the production bucket and the key prefix `''`; 16 `ok` lines, of which 15 end in `nothing` and the one for `private/ff/v1/miners` shows the count of miner files; the last line `PASS  no object at a key that v2 writes`; `exit 0`. Not a pass:

   - A first line with `key prefix 'smoke.../'`: the `SN120_R2_PREFIX=` at the start of the line was not typed. Nothing was written; run the line again.
   - `EXISTS`, `FAIL <key or prefix>` or `FAIL open the bucket`: as above. After a `FAIL <key or prefix>` line the last line still says `FAIL objects exist at keys that v2 writes`. That wording is wrong for this case: the list call failed, and no object was found.

   Send the output to the coordinator. The early run does not replace the run of this step on the cutover day.

   After step 6, a second run finds v2's own genesis files. Then name them as expected, one `--allow` for each: `python -m sn120_v2.r2 --preflight --allow public/bases/ --allow public/masters/`. Their lines then say `N objects, all under --allow`.
6. **Genesis files** (57 GB of disk, 30 GB of RAM, no GPU). With the genesis directory of the smoke test, nothing is downloaded or built again; the files are uploaded to `public/bases/<base hash>/` and `public/masters/<master digest>/` of the production store, and each stored file is read back and compared.

   ```
   python -m sn120_v2.gpu.genesis --out <the genesis directory of the smoke test> --upload 2>&1 | tee $CUT/genesis.log
   export MASTER_DIGEST=$(tail -n 1 $CUT/genesis.log | python -c "import json,sys; print(json.load(sys.stdin)['master_digest'])")
   ```

   Check: `echo $MASTER_DIGEST` prints the same 64 characters as in the smoke test.
7. **Import, real run.** Before it, step 4 of "AffineValidator" is done: v1's site is saved. The import reads `https://affine.io/k1.json` and `https://affine.io/mining.json`, and the pull replaces `mining.json` later. Save `k1.json` once more, take a new chain snapshot (step 3), and run the lines below. The reason for the new snapshot: the import gives a queue of about 256 waiting seats (the dry run of 13 Nov 2026 at 23:04 UTC; SN120 itself has 1000 registrations). With that queue, the first pass of `run` refuses the chain sync when 65 or more seats leave in one read, and that needs a snapshot that is more than about a day old (the history in `tests/data/cutover/k1.json` has 61 registrations at most in any 24 hours). At the bootstrap itself this refusal cannot come: the queue is empty before the sync of `--chain`. "When `run` refuses the chain sync" in "After the start" has the way out.

   ```
   python -m sn120_v2.cutover --out $CUT/import --chain $CUT/chain_snapshot.json \
       --master-digest $MASTER_DIGEST --validator-hotkey $HK --owner-hotkey $HK \
       --earlier-k1 tests/data/cutover/k1.json --earlier-k1 <each later copy>
   ```

   Check: no `CUTOVER IMPORT REFUSED`; the kings and the front seats are those of step 4; one `WROTE` line for each file. The command never replaces a directory: for a second run, use a new one. If v1's publisher of `k1.json` has stopped, the command refuses a file older than one hour: add `--max-age-s` with the smallest value that accepts it.

   **Once the site serves v2's `mining.json`, the import with the two public URLs is refused**, in the dry run too:

   ```
   CUTOVER IMPORT REFUSED (CutoverError): mining.json and k1.json are not from the same epoch
   ```

   In the order of this file that comes only after the pull's first run ("AffineValidator", step 12). Then give the command saved copies of v1's two files: add `--k1 <file>`, `--mining <file>` and `--max-age-s <seconds>` ("Options of the import command"). The copies are the two files of v1's site in `$BACKUP` ("AffineValidator", step 4), or the pair that you saved last ("The state of v1"). They are public files: copy them to this box.

   - The two files must be from the same time. The epoch of `mining.json` must be the last epoch of `k1.json`, or at most 2 epochs after it. So a new `k1.json` from the site with a saved `mining.json` gets the same refusal.
   - Without `--max-age-s` the refusal is `k1.json is N s old (limit 3600 s): refusing a stale read`. Give the smallest value above that N.
   - Give as `--earlier-k1` only copies that are older than the `--k1` file.
   - The import then carries v1's state at the time of the copies. A turn that v1 took after them is not in it. So v1's round path is stopped before the copies are made ("The state of v1", points 1 and 2). If v1 took a turn after them, do not go on: tell the coordinator.
8. **First start.** `--stack sn120_v2.assembly:stack` names the builder of the real parts. The command pins the parameters, writes the genesis head, the kings and the first queue, and publishes `llms.txt` and `mining.json` into the store. It refuses a second run on a journal that has a genesis.

   ```
   python -m sn120_v2 bootstrap --state-dir $STATE --params $CUT/params.json --stack sn120_v2.assembly:stack \
       --genesis $CUT/import/genesis.json --genesis-files <the genesis directory>/genesis_files.json \
       --v1-import $CUT/import/v1_import.json --chain $CUT/import/chain.json
   python -m sn120_v2 status --state-dir $STATE --stack sn120_v2.assembly:stack
   ```

   Check: `bootstrap` ends with exit code 0 and prints no line `not finished:`. Its first line goes to stderr and is no fault: `checking every genesis file in the store against its sha256 (the weights are large: this takes minutes)`. `kings` holds the 10 UIDs of step 4, as a set: `bootstrap` and `status` print them newest first, the import oldest first and, in `first weights`, in the order of the UID. `queue` is the count of waiting seats of step 7; `served` is the count of registrations in the `served:` line of the import; `published` is `true`; `params_hash` is the hash that the check line of check 5 printed for `$CUT/params.json` (so `squat_min_wait_s` is pinned as 31536000); `queued_files` of `status` shows how many waiting seats have a valid file.

   - `bootstrap` also writes the head pointer, `public/head.json`. The first run of the pull on AffineValidator needs it (step 12 there).
   - A line `not finished: head pointer not published yet: <cause>` (exit code 1): the store did not take the pointer. The genesis is written; do not run `bootstrap` again. The first pass of `run` (step 10) writes the pointer.
   - A line `not finished: file check failed: <cause>` (exit code 1): our own checker fails. The genesis is written; do not run `bootstrap` again. Do not go on: see "When no round opens" below.
   - If `queued_files` has a count under `check_fault`, send that line to the coordinator, and each line that holds `the checker fails on this file:`. Such a file is skipped and `run` goes on, but each one is a bug of ours or a hostile file.
9. **Tell AffineValidator to go on with step 5 of its list.** The journal copy is now in the store. Steps 5 to 11 of that list do not wait for step 10 here. Step 12 of that list (the first run of the pull) comes after step 10 here: the site then shows a round loop that runs, and the head pointer is in the store in every case.
10. **Start the round loop, through the restart loop.** On mainnet only the restart loop starts `run`. Never start `run` alone: nothing would start it again after a crash or a reboot, and a restart by hand can count a failure for a seat. The loop is one script, `systemd/sn120-v2-run-loop.sh`. The unit `systemd/sn120-v2-run.service` starts the same script. The lines below are those of [SMOKE.md](SMOKE.md), "After a green test: the restart loop of `run`". One thing differs: the file of the variables is `$CUT/run.env`. "A crash or a restart of `run`" under "After the start" gives the rules.

    1. **Nothing of the smoke test runs on this box.**

       ```
       pgrep -af "sn120-v2-run-loop|sn120_v2 run"          # must print nothing
       systemctl is-enabled sn120-v2-run 2> /dev/null      # a box with systemd: must not print "enabled"
       ```

       - If a line shows a `run` that was started by hand (under tmux), stop it with Ctrl-C, then look at `nvidia-smi`: no GPU holds memory. The loop cannot stop the servers of such a `run`. If they still hold a GPU, the loop does not start `run` and ends with exit code 4.
       - If a line shows a loop, stop it with its stop line (step 10.4).
       - If the unit is enabled, it is the unit of the smoke trial, with the state directory of the smoke test: `sudo systemctl disable --now sn120-v2-run`. Form B below writes the unit again.

       **The digests of the box are the pinned ones.** The bootstrap pinned the digests of `$CUT/plan.json`. A checkout or an install line after the digest block can change those of the box:

       ```
       python -m sn120_v2.gpu.plan | diff - $CUT/plan.json; echo "exit=$?"     # must print only exit=0
       ```

       With any other output, do not start the loop: see "The digests of the box and the pinned digests" in "After the start".
    2. **Start the loop.** Use one form, never both. Run it in `v2/`, in the shell of step 2.

       **Form A: a box without systemd** (a rented container).

       ```
       setsid nohup systemd/sn120-v2-run-loop.sh "$(command -v python)" --state-dir $STATE --stack sn120_v2.assembly:stack > /dev/null 2>&1 < /dev/null &
       ```

       After a reboot of such a box nothing starts the loop, unless the start command of the box holds this line. If it does not, start the loop by hand after each reboot.

       **Form B: a box with systemd.** The unit starts the loop at each boot.

       ```
       RENV=$CUT/run.env
       (umask 077; env | grep '^SN120_' > $RENV)             # the variables of run, mode 600. It holds the store token.
       sed -n 's/=.*//p' $RENV                               # the names only, no value
       sed -e "/^[^#]/s|SN120_USER|$(id -un)|" -e "/^[^#]/s|SN120_ENV_FILE|$RENV|" -e "/^[^#]/s|SN120_PYTHON|$(command -v python)|" \
           -e "/^[^#]/s|SN120_V2_DIR|$PWD|" -e "/^[^#]/s|SN120_STATE_DIR|$STATE|" \
           systemd/sn120-v2-run.service | sudo tee /etc/systemd/system/sn120-v2-run.service > /dev/null
       grep -v '^#' /etc/systemd/system/sn120-v2-run.service | grep SN120_      # must print nothing: every blank is filled
       systemd-analyze verify /etc/systemd/system/sn120-v2-run.service
       sudo systemctl daemon-reload && sudo systemctl enable --now sn120-v2-run
       systemctl status sn120-v2-run                         # "active (running)"; after the loop gave up: "failed"
       ```

       Check of the third line, before you go on: the names are `SN120_WORK_DIR`, `SN120_IMAGE_DIGEST`, `SN120_R2_ENDPOINT`, `SN120_R2_BUCKET`, `SN120_R2_ACCESS_KEY_ID`, `SN120_R2_SECRET_ACCESS_KEY`, `SN120_PUBLIC_URL`, `SN120_LINKS_URL` and `SN120_V2_CHAIN_NETWORK`. `SN120_R2_PREFIX` is not there. If a name is missing, export it (step 2, or check 3 of "Before the mainnet bootstrap" for `SN120_IMAGE_DIGEST`) and run the second line again.

       The unit reads `$RENV` at each start. So each later change of an `SN120_` variable needs the second line again, then a stop and a start of the loop. Do not `cat` the file.
    3. **Look at the loop.**

       ```
       cat $SN120_WORK_DIR/logs/run-loop.status              # one line: what the loop does now, or why it stopped
       tail -f $SN120_WORK_DIR/logs/run.log                  # the output of run, and the lines "run-loop: ..."
       ```

       The log of `run` is `$SN120_WORK_DIR/logs/run.log`. It is not `$CUT/run.log`.

       Check, from a second shell with the same exports:

       - The status line says `running: run has pid N, start 1 of at most 5 with no finished round`. **Read this line once after each start of the loop.** If it says `waiting for the store or the chain since ...`, read the `<cause>` at its end: "A crash or a restart of `run`" says which causes do not go away.
       - `run.log` has the lines `run-loop: loop started (pid N): ...` and `run-loop: start 1 of at most 5 with no finished round between them (rounds closed so far: 0)`.
       - The lines of `run` after them are not `waiting: ...`. If they are, see "When no round opens" below.
       - `status` shows a `phase` other than `IDLE` after the first lease.
       - `python -m sn120_v2 round --state-dir $STATE --stack sn120_v2.assembly:stack` prints the summary when the first round has closed. The 100 minutes of the time limit are from the lease to the promotion, not to the close: with a winner the round closes after the fold ("The time limit of a round" and "What a closed round looks like" below).
    4. **The stop**, for each later step that says "stop the loop". `run` and its servers end, and nothing starts them again.

       ```
       kill $(cat $SN120_WORK_DIR/logs/run-loop.pid)         # form A
       sudo systemctl stop sn120-v2-run                      # form B. "sudo systemctl disable --now sn120-v2-run" also keeps it down after a reboot.
       ```

       To go on after a stop: the line of form A again, or `sudo systemctl start sn120-v2-run`.

       A reboot after a stop: with form B the enabled unit starts the loop at boot, also after `systemctl stop`. Only `disable --now` keeps it down. With form A nothing starts the loop.
11. **Records.** Put `$CUT/import`, `$CUT/params.json`, `$CUT/plan.json` and each saved `k1.json` with the v2 run records.

## AffineValidator: the order at the cutover

One list, in this order. Each step has its check. Do not start a step before the check of the step before it passes. One exception: step 11 can go before step 9 (step 11 says how). Step 4 must be done before step 7 of "GPU box" (the import's real run). Steps 5 to 13 need step 8 of "GPU box" (the bootstrap). Step 12 also needs step 10 of "GPU box" (the round loop runs).

**What can be done now, before the smoke test is green: steps 1, 2, 3 and 4, the chain check of step 6, and the file check 8.1.** None of them changes anything that is live: they start no unit, enable no timer, stop nothing of v1, and write nothing to the chain, to the store or into the site directory. Each one is marked **Can be done now** in the list.

| Step | What it leaves on the host | What to do again at the cutover |
| --- | --- | --- |
| 1 | A v2 checkout and its virtual environment. | Check out the cutover commit and run the install line again. Then run `git diff --stat <the commit of today> <the cutover commit> -- v2/systemd/sn120-v2-weights.service v2/systemd/sn120-v2-weights.timer v2/systemd/sn120-v2-links.service v2/systemd/sn120-v2-links.timer`: if it names a file, run the `sed` blocks of steps 2 and 3 again. Only these four unit files matter here. The loop script of the GPU box changes often and is not installed on this host. |
| 2 | The weights writer's credentials file, and two unit files that are not enabled. | Nothing, but see step 1. |
| 3 | Two unit files that are not enabled. It only reads v1's credentials file and the site directory. | Nothing, but see step 1. |
| 4 | A copy of v1's site and the frozen v1 history, both outside the site directory. | Make both again right before step 7 of "GPU box": `mv "$BACKUP" "$BACKUP-early"`, then the copy block of step 4. The way back uses the copy, so the copy must be fresh. |
| 6, the chain check only | The script file. It only reads the chain. | Nothing. Step 9 runs it again. |
| 8.1 | Nothing. It reads the mode and the names of the pull's credentials file. | Step 8 runs it again. |

**In each new shell:** set the session names and the helper `with_token` again, then run `cd "$REPO/v2" && source .venv/bin/activate`. Steps 5, 6 and 9 call `python` with no path, and only this line makes it the Python of the checkout. Without it the weights writer ends with `No module named sn120_v2`. Nothing is written then: run the line, then the command again.

Everything else waits. Step 5, the two dry runs of step 6 and step 7 read the journal copy in the production store, and that copy exists only after the bootstrap. From 8.3 on, each step changes something that is live.

The order follows Jacob's sequence: the new `llms.txt` first (step 8), then the weights and the stop of v1 (steps 9 and 10), then the website (step 12). Do step 11 right after step 10: until v2's links job has run, the seats that v2 serves first cannot upload a file (step 11 says why). The first run of the pull is the site switch, so the pull does not run before step 12.

The pull's credentials file exists already; step 8 checks it.

The workers' documents hold more detail on each job: `api-weights.md`, `api-links.md` and `api-site.md`. The commands here are the same.

Run every step as the Linux user that runs v1's `affine-ff.service`: that user can read v1's credentials file and owns the site directory. The weights writer runs as the same user, so that user must be able to read the hotkey file. Names for this session (full paths, no spaces):

```
RUNUSER=$(whoami)
REPO=$HOME/affine                          # the v2 checkout: a directory of its own, never a directory of v1. You set this path.
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
V1FILE=$KEEP/v1-history.json               # the frozen v1 history. You set this path.
MOVED=$KEEP/moved                          # where v1's stale files go after the site switch
PY=$REPO/v2/.venv/bin/python               # the Python of the checkout
mkdir -p "$KEEP"
```

The pull's credentials file has the store settings under the names `R2_ENDPOINT`, `R2_BUCKET`, `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`. Two commands read these names: the pull (`site/pull.py`) and `site/reset.py llms`. No other v2 command does. Use the file as it is: no copy and no second file. Run this block once; it defines a helper named `with_token` for a run by hand.

```
with_token() { env -i PATH="$PATH" HOME="$HOME" bash -c 'set -a; . "$0"; set +a; exec "$@"' "$PENV" "$@"; }
```

`with_token <command>` starts the command in a new process that has only `PATH`, `HOME` and the variables of the file.

- The token never enters your shell, and nothing prints it.
- No `SN120_*` variable of your shell reaches the command. This matters: a variable with `SN120_R2_` at its start wins over the plain name of the file, so a stray `SN120_R2_PREFIX` or `SN120_R2_BUCKET` of the smoke test would send the command to another place.
- Do not use `set -a; . "$PENV"` in your own shell, and do not `cat` the file.

The other commands of this list do not need the helper. Steps 5, 6 and 9 load the weights writer's file `$WENV`, which sets all five `SN120_R2_` names. The units of steps 9, 11 and 12 read only their own files. Only the rehearsal of step 7 reads your shell: it has its own check.

1. **Install.** **Can be done now**, with the newest commit of `v2`. At the cutover: the same commit as the GPU box.

   `$REPO` is a checkout for v2 alone. v1 runs from `/home/const/subnet120` and `/home/const/subnet120-rewrite`: never check out another commit there, and never install into v1's environment. If `$REPO` does not exist yet: `git clone https://github.com/AffineFoundation/affine "$REPO"`.

   ```
   cd "$REPO" && git fetch origin && git checkout <commit that the coordinator names>
   git merge-base --is-ancestor 3953282834931ece95dccfae720ea01049a1d1cd HEAD; echo "exit=$?"
   cd v2 && uv venv .venv --python 3.12 && source .venv/bin/activate
   uv pip install -e ".[chain,r2,links]"
   install -d -m 700 "$WSTATE" "$LSTATE"
   ```

   The commit is the one of the GPU box: `0c29cb6c`, or a later one that the coordinator names (check 1 of "Before the mainnet bootstrap").

   Check: the second line prints `exit=0`: the commit contains PR #139, so the pull and `reset.py llms` read the names of the pull's credentials file. `python -c "import bittensor, boto3, nacl, sn120_v2.weights_writer, sn120_v2.links; print('ok')"` prints `ok`.
2. **The weights writer: credentials file and unit files.** **Can be done now.** Write `$WENV` with these five lines, with no quotes and no `export`, then `chmod 600 "$WENV"`. The token is the read-only one.

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

3. **The links job: host check and unit files.** **Can be done now.** The job reads v1's credentials file in place. Never use `cat`, `grep` or `source` on that file.

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

4. **Check the host, copy v1's site, and freeze the v1 history.** **Can be done now**; make the copy and the history again right before step 7 of "GPU box" (the import's real run). Nothing here changes the site directory.

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

   Check: stderr says `kings: read from the journal copy in the store`. The JSON line has `"status": "would set"`; `weights` holds 10 UIDs at 0.1 each, and no UID 0; `validator_hotkey` is the hotkey above; `"validator_registered": true`. The parentheses keep the token out of the shell that the later steps use.

   Which 10 UIDs: before v2's first winner the line has `"kings": 10`, and the UIDs are the 10 kings of the import. After a winner of v2 they are not: step 9.1 says which UIDs are right then.
6. **Dry run with the wallet, and the chain check.** The dry run also reads the hotkey file and runs every check that comes before a submission. Still no chain write.

   ```
   ( set -a; . "$WENV"; set +a
     python -m sn120_v2.weights_writer --state-dir "$WSTATE" --from-store --network finney \
         --wallet "$WALLET" --wallet-path "$WALLETS" --dry-run )
   ```

   Check: `wallet_hotkey` equals `validator_hotkey`, and either `"preflight": "passed"`, or `"status": "would defer"` with the reason `recent write: the rate limit allows the next one in N blocks`. That reason is normal while v1's writer still writes. Any other reason, or `nothing submitted: the hotkey file ...`: stop and correct it.

   Save the chain check script: the block in "The chain check" below. It only reads the chain and needs no wallet. Step 9 runs it several times. **Can be done now**: the script and its run, not the dry run above. Then run it:

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

   Before the block, `env | grep -c '^SN120_R2_'` must print `0` (the command prints a count, never a value). The block sets the endpoint and the bucket itself, and the keys come from `$CRED`; any other `SN120_R2_` variable of your shell would win over them.

   Check: one line `links: issued N of M waiting hotkeys (not activated ..., not Ed25519 ...), removed 0, valid 86400 s`, exit code 0. `M` is the queue of the bootstrap. If the second count is most of v1's files, stop and ask: then the v2 queue is not what the import promised.
8. **The new `llms.txt` alone.** Only this one file goes up here: the `llms.txt` that the bootstrap published into the store. v1's `mining.json`, `index.html` and every other file of the site stay as they are. Do not run the pull for this: one run of the pull also replaces `mining.json` and `index.html`.

   1. **Check the pull's credentials file.** **Can be done now.** It exists already: `$PENV`. Do not make a new one, do not edit it, and do not show its content.

      ```
      stat -c '%a %U' "$PENV"
      sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' "$PENV"
      ```

      Check:

      - The first line is `600 const` (the mode and the run user).
      - The second command prints only the names, never a value: `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET` and `R2_ENDPOINT`, in any order.
      - If a name is missing from that print, its line does not start with the name (for example it starts with `export`). systemd does not read such a line, so the unit of step 12 would fail. Tell Jacob; do not edit the file yourself.
      - The print does not show `SN120_R2_PREFIX`: production has no prefix.

      The token can only read, and only this one bucket. It is not the credentials file of the upload links job: that one can sign uploads.
   2. **Check that the store's `llms.txt` comes from the cutover build.** `llms.txt` must not go up from a store that an older commit published.

      - The commit of the GPU box holds the nine commits of check 1 of "Before the mainnet bootstrap": the loop there printed `exit=0` nine times. Four of them are the reason for this step: the round time hard stop (`9b830458`), the limit of 3 checks (`5ce10ce7`), the corrected text of `llms.txt` (`4e23a275`) and its words on `no_decision` (`0c29cb6c`).
      - `0c29cb6c` is the least commit whose `llms.txt` may go up. The commit of the GPU box (`git rev-parse HEAD` there) is that commit, or a later one that the coordinator names.

      The text check. It reads the store's `llms.txt` with the command of step 8.4, into a new empty directory and not into the site directory:

      ```
      CHECK=$(mktemp -d)
      with_token "$PY" "$REPO/v2/site/reset.py" llms --site-dir "$CHECK"; echo "exit=$?"
      grep -c "A round with no result" "$CHECK/llms.txt"
      grep -E 'params_hash|lease_timeout_s|squat_min_wait_s|The lease ends' "$CHECK/llms.txt"
      ```

      - The second line prints `reset llms: ... is the store's public/llms.txt: N bytes, sha256 H` and `exit=0`. With `exit=1`, see the end of step 8.4.
      - **The count is `1`.** The text of `0c29cb6c` holds the line that starts with `- A round with no result: the box failed, not a seat.` one time. No commit before it writes these words. Checked on 14 Nov 2026 with one `mining.json`: the text of `0c29cb6c` gives 1, and the text of `f986ccb1` gives 0.
      - **The count is `0`: do not go on.** `grep -c` then ends with exit code 1; that is no fault of the command. The `run` process that published this file runs a commit before `0c29cb6c`. Do "A checkout only: a running box goes to a later commit" in "After the start" on the GPU box. Then run the four lines again.
      - The last line shows the values of this state. They are values of the pinned parameters, not fixed text. So compare them with the GPU box, not with the numbers of this file:
        - The two `params_hash` lines show the `"params_hash"` that `status` prints on the GPU box. With another hash, the file does not show the parameters that are pinned now: do not go on, and tell the coordinator.
        - The row of `lease_timeout_s` shows the pinned value in seconds. The line `The lease ends N minutes after it starts` shows that value divided by 60. On 14 Nov 2026: `7200` and `120`.
        - The row of `squat_min_wait_s` shows `31536000` (check 5 of "Before the mainnet bootstrap"). With `86400`, see the `squat_min_wait_s` line in the check of step 8.4.

      The same count on the public file, after step 8.4 and at any later time: `curl -s https://affine.io/llms.txt | grep -c "A round with no result"` prints `1`.
   3. **Stop the v1 unit that puts v1's `llms.txt` back.** `affine-network-dashboard.service` writes `network-data.json` into the site directory every 15 seconds. In the same pass it compares `llms.txt` and `reward-policy.json` with its own copies and writes its copies back if they differ. While it runs, the new `llms.txt` lives for 15 seconds at most.

      ```
      systemctl --user disable --now affine-network-dashboard.service
      ```

      After this, v1's page shows `network-data.json` as it was at the stop. This is the cost of the new `llms.txt` going first.

      Check: `systemctl --user is-active affine-network-dashboard.service` prints `inactive`. If no such unit exists, find the process: `pgrep -af dashboard.server`.
   4. **Copy the file.**

      ```
      with_token "$PY" "$REPO/v2/site/reset.py" llms --site-dir "$SITE"; echo "exit=$?"
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
      curl -s https://affine.io/llms.txt | grep -c "A round with no result"
      grep squat_min_wait_s "$SITE/llms.txt"
      sleep 60; sha256sum "$SITE/llms.txt"
      ```

      - The command printed `reset llms: ... is the store's public/llms.txt: N bytes, sha256 H` and `exit=0`.
      - Both `sha256sum` lines and the `curl ... | sha256sum` line show `H`.
      - `head` shows `# Affine (Bittensor SN120) v2: the miner contract`.
      - The title line shows a page title: the browser view works.
      - `grep -c` prints `0`: the root still shows v1's page.
      - The `Where to get the link` line names `https://affine.io/ff/links/<your hotkey>.json`. If it says `not published yet`, see step 13.
      - The count of `A round with no result` is `1`: the public file is the text of `0c29cb6c` or of a later commit (step 8.2). With `0`, the file of an older commit is up: do "A checkout only: a running box goes to a later commit" in "After the start" on the GPU box, then run the command of this point again.
      - The `squat_min_wait_s` line shows `31536000`. With `86400`, the bootstrap got a file without the key (check 5 of "Before the mainnet bootstrap"). Tell the coordinator at once: the repair is a `set-params` with the right file between two rounds, and it must come before the first entry is charged. Its `changed` must then show `"squat_min_wait_s": {"before": 86400, "after": 31536000}` and no other key.
      - After 60 seconds the hash is still `H`. If it changed, another v1 process writes `llms.txt`: v1's source speaks of "legacy website pushes". Find it (`pm2 ls`, `crontab -l`, `grep -rl llms ~/.config/systemd/user/`), stop it, and run the command again.

      With `exit=1`, the file in the site directory did not change. `missing in the store` means that the GPU box has not published yet, or that the bucket is wrong. `not of the same snapshot` means that a publish was in progress: run it again.

   To take only this step back (the new `llms.txt` is up, the site is still v1's): `cp -p "$BACKUP/llms.txt" "$SITE/llms.txt.tmp" && mv "$SITE/llms.txt.tmp" "$SITE/llms.txt"`, then start `affine-network-dashboard.service` again.
9. **Move the weights writer from v1 to v2. This is one step: do not pause inside it.** Only one writer can set weights: two writers on one hotkey block each other at the rate limit, and each overwrites the other. While v1's writer runs, every v2 run defers at the rate limit, so v1 stops first.

   **The clean window for the first write (9.4): minute 25 to minute 44 of an hour.** v1 writes at minute 01. The chain allows one write per 100 blocks for one hotkey (20 minutes), so from minute 21 the next write is allowed. v2's timer runs at minute 05 of each hour. After a first write up to about minute 44, 100 blocks or more lie before that run, and it writes.

   **A first write from minute 45 to minute 55 is also correct.** Then fewer than 100 blocks lie before the timer's run at minute 05. That run defers with `recent write: the rate limit allows the next one in N blocks`, and it uses the attempt of its hour. Nothing is lost: v2's first table stays on chain, and the next write comes one hour later (9.7). Do 9.2 before minute 55 in any case: v1's next run is still far then.

   SN120 has commit-reveal on: the weights go to the chain hidden (the commit), and the chain opens them by itself after one epoch, at most 72 minutes later (the reveal). So `set` means that the chain accepted the commit; the weights change at the reveal.

   1. **Check before the stop.** Run the chain check and the dry run with the wallet (step 6) again.

      Check: write down `last write block`; this is v1's last write. The dry run gives `"preflight": "passed"` or the reason `recent write`. Its `weights` hold 10 UIDs at 0.1 each. Compare these UIDs with the UIDs of `weights on chain`:

      - **Before v2's first winner** (the dry run has `"kings": 10`): the two sets are the same 10 UIDs, the kings of the import.
      - **After a winner of v2, the two sets differ, and that is correct.** `weights on chain` is v1's last write: v1 pays its own 10 kings and knows no winner of v2. v2 pays the last 10 kings: each winner of v2 with a new UID takes one place, and the oldest king of the import leaves.
        - The kings of the import, oldest first (the import printed them as `kings, oldest first`; these are the UIDs of 14 Nov 2026): 955, 956, 942, 821, 851, 909, 822, 852, 918, 920.
        - After one winner: the dry run has `"kings": 11`. Its `weights` hold the winner and 9 UIDs of the import. UID 955 is not there. After two winners: `"kings": 12`; UIDs 955 and 956 are not there. And so on down the list.
        - A winner whose UID is already one of the 10 takes no new place: `"kings"` is one more, and the UIDs do not change.
      - **How to check it.** On the GPU box, `python -m sn120_v2 status --state-dir $STATE` prints `"kings"`: the 10 paid UIDs, newest first. The UIDs of `weights` in the dry run are this list, as a set. Then:
        - each UID that is in the dry run and not on chain is a winner of v2: it is at the start of `"kings"`, and the `winner:` line of a round summary names it;
        - the UIDs that are on chain and not in the dry run are the same count, and they are the first UIDs of the list above, with no gap.
      - The site is late here: v2 writes `mining.json` at the close of the round, after the fold, and the journal copy that the writer reads has the winner from the promotion on. So during a fold the dry run shows the winner and the site does not yet. `status` on the GPU box is the truth.
      - **Any other difference: stop and ask the coordinator.** These are: a UID in the dry run that is no winner of v2 and no king of the import; a UID on chain that is no king of the import (v1 still takes turns, so v1 can make a king after the import); a king of the import that is missing while an older one is still there; fewer or more than 10 UIDs; a weight other than 0.1; UID 0.
   2. **Stop v1's writer.** "v1 units", group B, gives the candidate names; the host is the truth. For the timer and the service that the first two commands show:

      ```
      systemctl --user list-timers --all
      grep -rlE 'weights_flip|live_reward_writer|current_assessment_writer|set_weights' ~/.config/systemd/user/
      systemctl --user is-active <v1 writer>.service     # must print "inactive": do not stop a write that is in progress
      systemctl --user disable --now <v1 writer>.timer
      systemctl --user stop <v1 writer>.service
      ```

      Check: `systemctl --user list-timers --all` does not list the v1 writer's timer, and `systemctl --user is-enabled <v1 writer>.timer` prints `disabled`. `affine-hourly-burn.timer` and `affine-transition-weights.timer` stay disabled and masked. Never start them.
   3. **Wait for the rate limit, and for an epoch that does not end at once.** Run the chain check again.

      Check:

      - `last write block` is still the block of 9.1. If it moved, a v1 writer is still alive: go back to 9.2.
      - Wait until it prints `next write allowed in blocks: 0`. This is at most 20 minutes after v1's last write.
      - Wait until it prints `next epoch in blocks:` with a value of 10 or more. If the value is less than 10, wait 3 minutes and run the check again: then the epoch has changed and the value is large.

      Do not start v2 before this. A first run inside the rate limit submits nothing, and it uses the attempt of the hour. A commit in the last 3 blocks of an epoch can open one epoch (72 minutes) late.
   4. **First run of v2's writer.** This is the first real weight write of v2. The command returns when the run ends; a write waits about one minute for the block to be final.

      ```
      sudo systemctl start sn120-v2-weights.service
      journalctl -u sn120-v2-weights.service -n 20 --no-pager
      ```

      Check: the log shows `kings: read from the journal copy in the store`, then one JSON line with `"status":"set"`. With `"status":"deferred"`, the reason is in the line. It says if a submission went out:

      - **The reason starts with `submission failed:`, or it is a text of the chain** (none of the writer's own texts of the next point): a submission was sent, and the writer did not see the chain accept it. Run the chain check before any other command. If `last write block` is still the block of 9.1, the chain took nothing: correct the cause, then run one more attempt by hand. If `last write block` is new, the commit is on chain: go on with 9.5 and 9.6, run no second attempt, and send the line to the coordinator.
      - **Every other reason: nothing was submitted**, and v1's last weights stay on chain. These are the writer's own texts: `recent write: ...`, `plan failed: ...`, `plan refused: ...`, `this hour already had its attempt`, `another weights writer holds the lock`, `the writer state file is unreadable: ...`, `the weight setter failed: ...`, `the validator hotkey is not registered on the subnet`, `UIDs [...] are not on chain`, `the wallet does not sign as the validator hotkey`, and each text that ends with `nothing submitted` or with `nothing submitted, the last valid weights stay`. Correct the cause, then run one more attempt by hand.

      `--force` only skips the rule of one attempt for each clock hour; every other check stays.

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

      Check: `last write block` is a few blocks after the `block` of the `set` line, and a line `commit that waits for its reveal: block <the same block> | opens at <time>` is there. About 3 minutes after that time, run the chain check again: no `commit that waits` line has that block any more, and `weights on chain` equals the `weights` of the dry run. Compare the block, not the presence of a line: a later write puts a new `commit that waits` line there with another block. The timer's run at minute 05 is such a write.
   7. **Check the first timer run**, after the next minute 05: `journalctl -u sn120-v2-weights.service --since "70 minutes ago" --no-pager`.

      Check: the timer's run shows `"status":"set"`.

      - **`recent write: the rate limit allows the next one in N blocks` is normal after a first write (9.4) later than about minute 44.** Fewer than 100 blocks lie between the two. Nothing is lost: v2's first table stays on chain. The proof: the chain check shows the `last write block` of 9.6. The next write is one hour later: read the log again after the next minute 05, and that run shows `"status":"set"`.
      - `recent write` in any other case, or a `last write block` that moved while no v2 run shows `"status":"set"`: another writer wrote in the last 20 minutes. Find that writer and stop it (9.2).

   Go on with step 10 after the check of 9.5. The checks of 9.6 and 9.7 come later by the clock; do them when their time comes.
10. **Stop v1's writers of the site directory, then check that nothing writes.** v1's weight writer is already stopped (step 9). Stop and disable `affine-ff.service`, `affine-k1.service` and each other unit of group C in "v1 units" that the list of step 4 shows as active. Two writers of one public file are never allowed: a v1 unit that still runs writes `mining.json` or `llms.txt` over v2's files, and the pull writes them back a minute later, so the public files change back and forth.

    **Where v1's round path still runs (group A of "v1 units"), the order is: the watchdogs, then group A, then this step.** In the order of this runbook, group A is stopped before the import ("The state of v1", point 1), so this step starts at the first line of the block. The reason for the order is in v1's code in this repository; nobody read it from the host:

    - A watchdog that still runs can bring back a unit that this step stopped. The revert of `ff-watchdog` starts `affine-mimo-lifecycle.service` and runs `systemctl --user restart` on `affine-mining-discovery` and four more units. A restart also starts a unit that is stopped and disabled. `affine-mining-discovery` is v1's writer of `mining.json`, so v1's file would go over v2's. "v1 units", group A, point 1 has the detail.
    - The first line of the block lists the watchdogs. **The result to expect: no watchdog that reverts v1**, because each one has ended by itself. `affine-perf-watchdog` can be in the list: see the next point. If another watchdog is listed, stop and disable it before any other unit.
    - `affine-perf-watchdog.timer` runs every 5 minutes. It restarts `affine-perf-eval` and `affine-perf-eval-2` when they are not active and still enabled. So `stop` alone does not hold for these two units: use `disable --now` as in the block, or stop the timer first.

    ```
    systemctl --user list-units '*watchdog*'
    systemctl --user disable --now affine-ff.service affine-k1.service        # and each other unit of group C that is active
    touch "$KEEP/mark"; sleep 90
    find "$SITE" -newer "$KEEP/mark" -type f -not -path "$SITE/ff/links/*" -not -name links.json -not -path "$SITE/public/*"
    ```

    Check: for each unit, `systemctl --user is-active <unit>` prints `inactive` and `is-enabled` prints `disabled`. `issued_at` in `https://affine.io/ff/links.json` does not change any more, and `generated_at` in `https://affine.io/ff/status.json` does not change any more: v1's service writes both, and v2 writes no `ff/status.json`. The check on `ff/status.json` works only before step 12.3, because `reset.py out` moves that file out of the site directory: this is one reason why this step comes before step 12.3. `find` prints nothing. Each line that it prints is a file that some process still writes: find that process and stop it, then run the last two lines again.

    **`find` when v2's pull already runs** (step 12 was done before this step). The pull writes `mining.json`, `llms.txt` and `site-data.json` at the root of the site when `run` published a new snapshot, and only then. So `find` can print these three names, and the writer can be v2's own pull. The test: run the last two lines again; the three names must not come again. A second test for `mining.json`: `python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("epoch"))' "$SITE/mining.json"` prints `None` for v2's file. If it prints an epoch name of v1, `affine-mining-discovery.service` runs again. Any other name in the output of `find` is not from the pull.

    **What can start a stopped unit again** (from the repository, not from the host):

    - `systemctl --user stop` holds. No unit file in the repository has `OnFailure=`, `PartOf=`, `BindsTo=` or `Requires=`, and `Wants=` names only `network-online.target`. The repository does not hold the unit files of the lifecycle, of `affine-k1`, of `affine-mining-discovery`, of `affine-training-params`, of `affine-k1-transparency` and of the gate. For these the host is the truth: `systemctl --user cat <unit> | grep -E '^(Restart|OnFailure|PartOf|BindsTo|Requires|Wants)='`.
    - A kill is not a stop. `affine-ff.service`, `affine-perf-eval`, `affine-perf-eval-2`, `affine-bench` and `affine-fp32-master-publish` have `Restart=always`: systemd starts them again after a kill or a crash (`affine-ff.service` after 20 seconds), not after a stop. Never use `kill` on a unit.
    - `systemctl --user restart` starts a stopped unit. These scripts of v1 run it on `affine-ff`, `affine-mining-discovery`, `affine-training-params` and `affine-k1-transparency`: `activate_tp.sh`, `activate_sm.sh`, `activate_12k.sh` (the switch and the revert of each) and `tools/caps/caps_apply.py`. Run none of them after this step.
    - The public sign that `affine-ff.service` is back: `generated_at` in `https://affine.io/ff/status.json` moves again. After step 12.3 the sign is that `$SITE/ff/status.json` is there again: step 12.3 moved the old file away.

    After the stop of `affine-k1.service`, `k1.json` stands still and proves nothing about v1's lifecycle: "The state of v1", point 2, has the check of the unit.
11. **First run of the links job, then its timer.** Two conditions: `affine-ff.service` is stopped, and the bootstrap is done. See "Upload links job" for the reason.

    **This step needs the stop of `affine-ff.service` only. It does not need step 9, and it does not need the other stops of step 10.** So when step 9 waits for its minute, do this step first:

    1. The first line of the block of step 10: no watchdog that reverts v1 is listed.
    2. `systemctl --user disable --now affine-ff.service`, with the check of step 10 for this one unit: `is-active` prints `inactive`, `is-enabled` prints `disabled`, and `generated_at` in `https://affine.io/ff/status.json` does not change any more.
    3. The block below. Then step 9, then the rest of step 10.

    Why this is safe, from v1's code in this repository:

    - v1's weight writer does not need `affine-ff.service`. By the text of v1's K1 service (`ops/k1_pass_rates/services/k1_service.py`), the hourly writer reads `website/scores.json`, and `affine-k1.service` writes that file from v1's turns and verdicts. Step 9 stops the writer, and step 10 stops `affine-k1.service` after it.
    - The stop of `affine-ff.service` alone starts no revert: `ff-watchdog` reads no file of that service.
    - The stop does not end v1's turns either. v1's lifecycle takes the hotkeys from the chain file of `affine-k1.service` and lists the miners' files in the bucket by itself. A miner's file has the same key for v1 and for v2. So while the lifecycle runs, a file that is valid for v1 gets a v1 turn. v1 only copies the file: the seat in v2 is not touched. The stop of the lifecycle ends the turns ("v1 units", group A).

    Do not wait with this step. v1's service gives no link to a hotkey that had its v1 turn. The front seats of v2 are such hotkeys: the v1 turns with no verdict ("What the import carries"). So while v1's service issues the links, a front seat has no link file (HTTP 404 at `https://affine.io/ff/links/<hotkey>.json`) and cannot upload a new file. It needs a new file when its v1 file has train keys that only v1 had: v2 takes that file as invalid ("What the import carries"). The other seats are not blocked in the meantime: the link that v1 issued points to the same object that v2 reads.

    The numbers of a read on 14 Nov 2026, 01:13 UTC, with v1's service as the writer: the index named 209 hotkeys. v2's job would issue a link file for each of them, and for the hotkeys that v1 gives no link: the 14 front seats, and each hotkey that had a v1 turn after the import. At that read this was UID 893 (its v1 turn at 00:38 UTC); UID 876 came at 01:20:51 UTC. **This count is not fixed.** While v1's lifecycle runs, each v1 turn adds one hotkey that v1 gives no more link and that keeps its seat in v2. v2 found no valid file for a front seat.

    ```
    sudo systemctl start sn120-v2-links.service
    journalctl -u sn120-v2-links.service -n 5 --no-pager
    curl -s https://affine.io/ff/links.json | head -c 400
    sudo systemctl enable --now sn120-v2-links.timer
    systemctl list-timers sn120-v2-links.timer --no-pager
    ```

    Check: the log shows the `links: issued ...` line of step 7; `removed` is now the second count of step 7. In `ff/links.json`, `issued_at` is new and `link_file` is `https://affine.io/ff/links/<hotkey>.json`. The key `link_file` is the public sign of the writer: only v2's job writes it, and the index of v1's service has no such key. One link opens with a test hotkey that you own: `python -m sn120_v2.links open --hotkey <address> --key <key file> --links-url https://affine.io/ff/links`. The timer is listed (minute 10 and minute 40 of each hour).
12. **The website: the first run of the pull is the site switch.** One run copies v2's public files from the store into the site directory, writes `public/read_links.json` (signed read links for the weight files of the current head, new on every run), and builds the status site. It never writes to the store, it never copies a weight file, and it takes only keys that v2's code names, never everything under `public/`. See "Public pull and site".

    Once the store holds `public/mining.json` and `public/llms.txt`, a run replaces these files at the root of the site, with no check of who wrote them: `llms.txt`, `mining.json`, `index.html`, `v1.html`, `site.js`, `site.css`, `favicon.svg`, `network-haffer.ttf`, `network-mono.ttf`, `site-data.json`, `v1-data.json`. So no run of the pull, by hand or by the timer, starts before all of these:

    - step 4: the copy and the frozen history exist;
    - step 7 of "GPU box": the import has read v1's `mining.json` from `https://affine.io/mining.json`;
    - step 10: nothing of v1 writes into the site directory;
    - step 10 of "GPU box": the round loop runs, and the bootstrap printed no line `not finished:`.

    The order inside this step: the pull first, then v1's files move out. Before the pull, v1's page still needs those files, and the reset tool refuses to move them until the v2 page is up.

    1. **One run of the pull by hand.**

       ```
       with_token "$PY" "$REPO/v2/site/pull.py" --site-dir "$SITE" --public-url https://affine.io --v1-history "$V1FILE"; echo "exit=$?"
       ```

       - The run writes the data files first (`mining.json`, `llms.txt`, the files under `public/`), then the fonts, the script and the style, and `index.html` and `v1.html` last. So no page is up before the files that it loads.
       - Do not run `build.py` by hand for the switch. Without the pull's data it writes a page that says "No data yet."

       Check on the host: the run prints `pull: wrote N, unchanged N, removed N, failed steps 0; site built` and `exit=0`. With `exit=1`, each failed step is one line `pull: FAILED ...` on stderr; before the GPU box has published, no root file was replaced, v1's page is still up, and you can run it again.

       One failed step is known: `pull: FAILED head and read links: missing in the store: 'public/head.json'`, with `exit=1`. The store has no head pointer yet. The other steps ran: the v2 page is up, but `public/read_links.json` is missing, and `llms.txt` names that file. The bootstrap writes the pointer, and the first pass of `run` writes it if the bootstrap could not (step 8 of "GPU box"). So make sure that step 10 of "GPU box" runs, wait for its first line `idle: ...` or its first lease, and run the pull again: it then prints `failed steps 0`. Do not install the timer before that.

       If the run stops with `the store setting ... is not set`, the message names the variable. If it stops with `R2_ENDPOINT is not a store endpoint`, the value in the file has a form that the command refuses: tell the coordinator the form, not the value.

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
       grep -E '^(User|EnvironmentFile|ExecStart)=' /etc/systemd/system/sn120-v2-site-pull.service
       sudo systemctl daemon-reload
       sudo systemctl start sn120-v2-site-pull.service; echo "exit=$?"
       journalctl -u sn120-v2-site-pull.service -n 20 --no-pager
       sudo systemctl enable --now sn120-v2-site-pull.timer
       ```

       The three lines that `grep` prints, with the session names as this file sets them:

       ```
       User=const
       EnvironmentFile=/home/const/sn120-v2/secrets/r2-public-pull.env
       ExecStart=/home/const/affine/v2/.venv/bin/python /home/const/affine/v2/site/pull.py --site-dir /home/const/subnet120/affine/website --public-url https://affine.io --v1-history /home/const/sn120-v1-archive/v1-history.json
       ```

       Two paths in the last line are values that you set, not facts of the host: the checkout (`/home/const/affine`, from `$REPO`) and the history file (`/home/const/sn120-v1-archive/v1-history.json`, from `$V1FILE`). If you set `REPO` or `V1FILE` to another path, the line shows your path. The user, the token file and the site directory are Jacob's facts.

       Check:

       - `systemctl start` on the service makes one run through the unit and waits for it. It prints `exit=0`, and the journal shows the same line `pull: wrote ... failed steps 0; site built`. This proves that systemd reads the token file. With another exit code, do not enable the timer: read the journal lines.
       - Enable the timer, not the service. `systemctl list-timers sn120-v2-site-pull.timer --no-pager` lists the timer. After two minutes, `journalctl -u sn120-v2-site-pull.service -n 5 --no-pager` shows `failed steps 0`.

       The unit loads `$PENV` as it is. If the log shows `the store setting SN120_R2_ENDPOINT is not set`, the checkout is older than the cutover commit: disable the timer (`sudo systemctl disable --now sn120-v2-site-pull.timer`), tell the coordinator, and do not edit the credentials file. The site stays as the run by hand left it.
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

    Check: the line names `https://affine.io/ff/links/<your hotkey>.json`. If it says `not published yet`, the GPU box runs without `SN120_LINKS_URL`: stop the loop there, set the variable (with form B of step 10 of "GPU box", also write `$CUT/run.env` again), start the loop, and do not announce before this check passes.

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
| `next epoch in blocks` | How many blocks are left in the epoch. The first v2 write starts only with 10 or more (step 9.3). |
| `commit that waits for its reveal` | A write that the chain has accepted and not opened yet. |
| `weights on chain` | The weights that count now. |

## What only the real host can prove

The code and the steps were tested on other machines, with a stand-in for the web server and a local store. Each line below is a fact that only AffineValidator, the real bucket or the edge can show. Jacob runs the check at the step that is named. A check that fails stops the list at that step.

| # | The fact | The check | When the check fails |
| --- | --- | --- | --- |
| 1 | The user of this list owns the site directory and can write there. | Step 4: `stat` shows `$RUNUSER` for the four paths, and `ps` shows `$RUNUSER` for `affine-ff.service`. | Do not change an owner or a mode in the site directory. Log in as the user that `stat` shows, and start the list again from the session names as that user. |
| 2 | `$KEEP` and the site directory are on one disk. | Step 4: the first `df` prints one mount point two times. | Set `MOVED` to a directory on the mount point of `$SITE`, outside `$SITE`. The copy `$BACKUP` can stay on the other disk. |
| 3 | v1 has no `public/` directory in the site directory. | Step 4: `ls -d "$SITE/public"` says `No such file or directory`. | Tell the coordinator before step 12. Do not run the pull: it refuses to replace a file there that it did not write. |
| 4 | The web server gives 404 as JSON for a link file that does not exist. | Step 4: the `curl` line prints `404 application/json`. | Tell the coordinator before step 11. Do not change the web server. |
| 5 | The pull's token reads the real store. | Step 8.4: `reset.py llms` prints `reset llms: ...` and `exit=0`. It is the first read with this token. | `missing in the store`: the GPU box has not published, or the bucket in the file is wrong. An access error: the token cannot read this bucket; correct the token, and never use v1's credentials file for the pull. Nothing in the site directory changed. |
| 6 | No other v1 process puts `llms.txt` back. | Step 8.4: after 60 seconds the hash is still `H`. | Find the process (`pm2 ls`, `crontab -l`, `grep -rl llms ~/.config/systemd/user/`), stop it, and run the command of 8.4 again. |
| 7 | The edge and the reverse proxy pass `llms.txt` with no cache. | Step 8.4: `curl -fsS https://affine.io/llms.txt \| sha256sum` shows `H`. | Run `curl -sI https://affine.io/llms.txt` and read `cf-cache-status` and `age`. If a cache holds v1's file, wait until it shows `H`, and tell the coordinator. Do not go on to step 9 before it shows `H`. |
| 8 | The validator hotkey file is readable for the writer's user, and the chain uses commit-reveal. | Step 6: `wallet_hotkey` equals `validator_hotkey`; the chain check prints `commit-reveal on: True`. | `nothing submitted: the hotkey file ...`: correct the path or the wallet name in step 2, and run the dry run again. `commit-reveal on: False`: stop and tell the coordinator. |
| 9 | The name of v1's weight writer unit. | Step 9.2: the two commands show its timer and its service. Step 9.3: `last write block` does not move. | If `last write block` moved, a writer is still alive: go back to 9.2. Do not start v2's writer before it stands still. |
| 10 | Which v1 units write into the site directory. | Step 10: `find` prints nothing after 90 seconds. | Each line is a file that a process still writes. Find the process, stop it, and run the last two lines of step 10 again. Do not start step 11 or step 12 before `find` prints nothing. |
| 11 | The real web server serves the new `public/` directory and `v1.html` with no restart, and the edge passes the new paths. | Step 12.1: every line of the two loops from outside starts with `200`. | A file is on the disk and its address gives 404: stop, and tell the coordinator. Do not restart the web server on your own. If the root page is broken, do "Go back". v1's files have not moved at this point. |
| 12 | No cache holds `public/read_links.json` for longer than a few minutes. | Step 12.1: `curl -sI https://affine.io/public/read_links.json`, the headers `cache-control` and `age`. | Tell the coordinator. The site can stay up; the read links are not announced before this is corrected. |
| 13 | The files of v1's real directory that no v1 page links. | Step 12.3: the dry run of `reset.py out` names nothing that must stay. | Do not do the real run. Send the dry run to the coordinator. The site works with v1's files still in place. |
| 14 | No object of the real bucket is at a key that v2 writes. | "GPU box", step 5: the preflight prints `PASS`. | Stop the cutover and send the output to the coordinator. Do not delete anything. |
| 15 | What the edge keeps for v1's `.js?v=` addresses. | Only after the `--assets` move and a later way back: "Go back", step 6, shows v1's page with its charts. | The edge can keep a "not found" answer for up to one day. So do the `--assets` move only when the way back is no longer needed. If it happened, tell the coordinator. |
| 17 | The restart loop starts `run` on the real box, with the production variables. The unit has run under no systemd yet, and the loop has stopped no real vLLM server and read no real `nvidia-smi` yet. | "GPU box", step 10.3: the status line says `running: run has pid N, start 1 of at most 5 ...`. | `run-loop.status` and the last lines of `run.log` say why, and what to run. Exit code 64 of the loop is a wrong call: a missing state directory or no `SN120_WORK_DIR`. Nothing public changes while `run` does not run: the head and the weights stay. |
| 16 | systemd reads the pull's token file, with the real user and paths. | Step 12.2: `sudo systemctl start sn120-v2-site-pull.service` prints `exit=0`, and the journal shows `failed steps 0`. | Do not enable the timer. If the journal shows `the store setting ... is not set`, a line of the file does not start with its name (step 8.1): tell Jacob. The site stays as the run by hand left it. |

## After the start

On the GPU box, in the shell of step 2 of "GPU box".

| Command | What it does |
| --- | --- |
| `python -m sn120_v2 status --state-dir $STATE` | The summary from the local journal: head, kings, queue, phase, `params_hash`, and two keys for the seats that a fault of the box touched: `"held_seats"` (the list of the UIDs of the held seats; `[]` when no seat is held) and `"seats_with_counted_failures"` (the number of seats that have a counted failure and are not held). Read-only. |
| `python -m sn120_v2 status --state-dir $STATE --stack sn120_v2.assembly:stack` | The same, read through the store, plus `queued_files`: the count of queued files for each admission result. |
| `python -m sn120_v2 round --state-dir $STATE --stack sn120_v2.assembly:stack` | The summary of the last closed round, as a text to paste. `--round r000007` picks one; `--json` prints the stored record. |
| `python -m sn120_v2 release-seat --state-dir $STATE --stack sn120_v2.assembly:stack --all` | Releases every held seat and sets the count of failures of every seat back to 0. `--uid 12` does it for one UID. Stop the loop first: this command takes over the journal. See "After a fault of the box in training". |
| `python -m sn120_v2 force-sync --state-dir $STATE --stack sn120_v2.assembly:stack` | Shows the chain sync that `run` refuses: the seats that leave. It writes nothing and is safe while the loop runs. With `--confirm-remove N` it applies the sync; stop the loop before that. See "When `run` refuses the chain sync". |
| `python -m sn120_v2 publish --state-dir $STATE --stack sn120_v2.assembly:stack` | Writes `llms.txt` and `mining.json` again. It does not change the journal. |
| `python -m sn120_v2 set-params --state-dir $STATE --stack sn120_v2.assembly:stack --params <file>` | Pins new parameters and prints the old and the new `params_hash`. From `6baf0d74` it also prints `changed`: each parameter whose value differs, with the value `before` and `after`. Read `changed` after each `set-params` ("The value `squat_min_wait_s`"). Stop the loop first: this command takes over the journal, and a `run` that still runs stops at its next write. Then start the loop again. The file holds the whole set of overrides, not only the change: a key that the file does not hold goes back to its default. So each file holds `"squat_min_wait_s": 31536000` ("The value `squat_min_wait_s`"). |

`--params` on `run`, `status` and `publish` is only a check: a file that differs from the pinned parameters is refused.

On AffineValidator: `journalctl -u sn120-v2-weights.service --since "3 hours ago" --no-pager` shows the last runs. The dry run of step 5 of "AffineValidator" shows `blocks_since_last_write`; one hour is 300 blocks, so it stays below about 310 while each hourly run sets. If the first log line of a run says `read from the cache of the journal copy`, the store did not answer or the store settings are wrong: the writer then pays the last kings that it read.

### What a closed round looks like

`run` prints the round summary when a round closes, and `python -m sn120_v2 round --state-dir $STATE --stack sn120_v2.assembly:stack` prints it again. [SMOKE.md](SMOKE.md) has the whole text with its table. The first three lines say what happened; `<...>` stands for a value.

**A winner.**

```
SN120 v2 round r000001: APPLIED
winner: UID <uid> (<hotkey>), delta +0.<4 digits>, z <number>
head: 9fa81073eeae -> <the first 12 characters of the new head>
```

- The head before the arrow is the head of the round. For the first round it is the genesis head (`9fa81073eeae` for the state of 14 Nov 2026). The head after the arrow is the folded head.
- `status` after the first winner: `"heads": 3` (the genesis head, the head with the winner's LoRA, the folded head), `"head_folded": true`, `"rounds_closed": 1`, `"last_outcomes": ["applied"]`, and the winner's UID first in `"kings"`. `"served"` is higher by the count of the charged seats, and `"queue"` is lower by the same count.
- The round closes after the fold, so the summary comes later than the promotion ("The time limit of a round").
- The weights: the next run of v2's writer pays the winner and no longer pays the oldest king (step 9.1 of "AffineValidator").

**No winner, with a scored batch.**

```
SN120 v2 round r000001: NULL
no winner: <the reason>
head: 9fa81073eeae (unchanged)
```

- Each scored batch is charged, win or lose: its seat leaves the queue (`"served"` rises and `"queue"` falls by that count). A batch with the verdict `not_trained` or `not_scored` is not charged and is first in the next round.
- `status`: `"heads"` and `"kings"` do not change, and `"last_outcomes"` ends with `"null"`.
- There is no wait after such a round: when valid files wait, `run` opens the next round at once.

**No decision.** The first line ends with `NO_DECISION`. Nobody is charged, and the seats are first in the next round. "A round in which no training finished" and "A crash or a restart of `run`" have the `no winner:` texts.

### The time limit of a round

A round has a time limit of 100 minutes from the lease to the promotion. It is a hard stop.

| Minute after the lease | What happens |
| --- | --- |
| 40 at the latest | The training time ends. A batch that is not trained then has the verdict `not_trained`. |
| 90 | The box stops scoring. A block of tasks that still runs is stopped and is not used. |
| 100 | The limit. The 10 minutes after the stop are for the grading of the last block, the judging and the promotion. |
| 120 | The lease ends (`lease_timeout_s` 7200; it was 14400). A result that arrives later is not used. |

**With a winner, the round does not close at minute 100.** The limit is from the lease to the promotion. `fold_every_k` is 1, so each winner is folded inside its round. The order is: judging, promotion, fold, then the close. `run` prints the round summary and writes `mining.json` only at the close. So during the fold:

- `status` on the GPU box shows `"phase": "APPLY"` and `"heads": 2` after the first winner; the winner is already first in `"kings"`.
- The site still shows `"phase": "RUNNING"` and the open round, and its `kings` are still the old ones. This is no fault.
- The code has no time limit for the fold. A fold that fails is tried again 5 minutes later (`apply_retry_s` 300); `run` then prints `idle: N seats wait, phase APPLY, last problem: fold failed: <cause>`.
- The stop that the code makes is the stop of the scoring at minute 90. The 10 minutes after it are a reserve; no timer ends the steps in them.

- A LoRA that is still open at the stop is judged on the blocks that null M and every open LoRA finished. It is charged, win or lose. The first line of the round summary then ends with `(stopped at the time limit)`.
- If fewer tasks than the first look are scored at the stop (500 tasks, a quarter of the 2,000 of a round), every trained LoRA has the verdict `not_scored`. It is not charged, it is first in the next round, and no failure is counted for its seat. The round summary then has the line `not scored: N of M batches. The time limit of the round came before the first look. They are not charged and are first in the next round.`
- A round with `not_scored` rows ends the same way the next time: the same seats come first. Send the coordinator the `wall time` line of the summary and the `eval` lines of `phases.jsonl`.

### When no round opens: the lines `waiting:`

`run` prints `waiting: <cause>` and then `idle: N seats wait, phase IDLE` at each try, every 30 seconds, and opens no round. The word `waiting:` reads as harmless. It is not: **no round opens until the cause is repaired**, and `run` does not end by itself. The restart loop does not help here: `run` does not end, so the loop does not start it again, and `run-loop.status` still says `running`.

There are four kinds of cause:

1. **The store did not answer** (`r2 <action> '<key>' failed: ...`). The lines stop by themselves at a later try.
2. **The chain did not answer** (`no connection to the chain at ...`, `chain read of ... failed: ...`, `chain read of the registrations did not end in 90 s: the connection is dropped`, `the chain head is N s old: the node is behind`, `partial chain read ...`). The lines stop by themselves when the chain answers again. **Do not stop the loop for a chain line: a restart does not help.** From `e9958463` (check 1 tests it with `f986ccb1`), one chain read of `run` ends in 90 s at most. A read that fails or does not end drops the connection, `run` prints `waiting:`, and the next pass opens a new connection. One pass waits for the chain 180 s at most (90 s for the registrations, 90 s for the block hash). A normal read of the registrations takes 5 to 8 s. What was tried: reads through a local relay to finney, with faults put into the relay. The chain library (bittensor 11.0.2) connected again in every case. What is not proven: a real outage on the box. If the same chain line still comes after 15 minutes, send the line to the coordinator.
3. **`run` refuses the chain sync** (`sync would remove N of M seats; refusing without force (possible bad chain read). If the chain is right, an operator applies it with the force-sync command`). Nothing ends it by itself, and a new start does not help. The way out is the command `force-sync`: see "When `run` refuses the chain sync" below.
4. **Our own checker fails its known-good case**, on this machine. Then nothing ends it, and a new start does not help. These are the texts of that case:

| The cause in the line | What fails |
| --- | --- |
| `the tokenizer fails on its known-good case: <error class>` | The tokenizer of the box. |
| `the official grader failed on a known-good case: <error class>`, `the official grader gives no known-good case its reward`, `the official grader failed: <error class>` | The grader of the math environment. |
| `the checker fails on a file (<place>) and does not pass its known-good file (<problem>)` | The file check as a whole. |

In the same state:

- `status` (with `--stack`) and `publish` end with `refused: InfraError: <cause>` and exit code 2.
- `bootstrap` prints `not finished: file check failed: <cause>` and ends with exit code 1.

What to do for kind 4: when the same `waiting:` line comes three times in a row, send the line to the coordinator. Do not delete a miner file and do not change the parameters to make a round open. A new start of `run` on the same machine gives the same line.

This is not the case of one bad miner file. A file that makes the checker fail while the known-good case passes is skipped with the rule `check_fault`, and `run` goes on. The same holds for `tokenizer_fault`, `grader_fault` and `grader_time_limit`. Such a file is checked again 10 minutes later at the earliest; after 3 counted checks its verdict stays until the miner uploads a new file. Send the coordinator the `queued_files` line of `status` whenever it has a count under `check_fault`, and each line that holds `the checker fails on this file:`.

### When `run` refuses the chain sync: `force-sync`

Words used here. A **seat** is one registration (UID and registration block) that waits in the queue for a round. The **chain sync**: at each pass, `run` reads the registrations from the chain; seats that are no longer on the chain leave the queue, and new registrations enter it.

The rule: `run` refuses a sync when the queue has 8 or more seats and more than 25 % of them leave in one chain read. The queue is the seats that wait: `"queue"` of `status`. So the limit follows the queue of that moment: more than a quarter of the seats that wait then. Such a read can be a bad chain read, so `run` itself never forces a sync. The command `force-sync` is there from `d32cd947` (check 1 of "Before the mainnet bootstrap" tests it with `f986ccb1`).

**Two numbers that are not the same.** SN120 has 1000 registrations on the chain (the chain values `SubnetworkN` and `MaxAllowedUids` were both 1000 at block 9501162). UID 0 has no seat. The import dry run of 13 Nov 2026 at 23:04 UTC gave 256 waiting seats (14 front, 47 normal, 195 back): this is the queue after the import, not the size of the subnet. With a queue of 256 seats, the refusal comes when 65 or more seats leave in one read. With a shorter queue the limit is lower.

When the refusal can come:

- At the bootstrap: never. The queue is empty before the sync of `--chain`.
- At the first pass of `run`: only if the chain snapshot of the bootstrap is more than about a day old (step 7 of "GPU box").
- In a normal round with a large queue: not in practice. Two reads are at most about 2 hours apart, and the history has 19 registrations at most in any 2 hours.
- **After a stop of `run` of more than about a day.**
- **When the queue is small:** with 8 to 11 seats, 3 seats that leave in one read are refused.

What you see before: `run` does not stop. It prints the same line at every pass (every 30 seconds), and no round opens:

```
tail -n 3 $SN120_WORK_DIR/logs/run.log
waiting: sync would remove 80 of 256 seats; refusing without force (possible bad chain read). If the chain is right, an operator applies it with the force-sync command
```

The loop status (`$SN120_WORK_DIR/logs/run-loop.status`) still says `running: run has pid N, ...`, because `run` did not stop. `status` shows the old queue size.

The command has two guards. It reads the registrations two times, at two blocks, and the two reads must give the same changes. And it applies the sync only when exactly the number of seats that you give leaves.

1. **Look at the list.** This writes nothing: the command reads the journal copy in the store and takes over nothing. It is safe while the loop runs.

   ```
   python -m sn120_v2 force-sync --state-dir $STATE --stack sn120_v2.assembly:stack > $CUT/force-sync-list.txt; echo "exit=$?"
   head -n 2 $CUT/force-sync-list.txt; tail -n 1 $CUT/force-sync-list.txt
   grep -c "remove uid" $CUT/force-sync-list.txt; grep -c WARNING $CUT/force-sync-list.txt
   ```

   The command takes about 21 s: one read of the 1000 registrations takes 5.6 to 7.0 s, and the second read is at the next block. The list is long, so the first line keeps it in a file: it has one line for each seat that leaves (333 seats gave 336 lines), and only the count of the seats that enter.

   The file (exit code 0; the block numbers and the counts here are examples):

   ```
   chain blocks 9500410 and 9500411 (1000 registrations) give the same changes to the queue of 256 seats:
     remove 80 seats, add 83 seats, change the hotkey of 0 seats
     remove uid   41 registration block   9412345 5F... (normal)
     remove uid   42 registration block   9412399 5G... (back)
     ... one line for each seat that leaves ...
   nothing was changed. To apply exactly this, stop run and give the same command with --confirm-remove 80
   ```

   In each line: the UID, the registration block, the hotkey, and the part of the queue (`front`, `normal` or `back`; `, held` when the seat is held).

   Check before step 2:

   - The first line says `(1000 registrations)`: the number of registrations is the size of the subnet, and SN120 has 1000. This is the right number. Another number means another subnet or another network: do not go on, and check `SN120_V2_CHAIN_NETWORK`. A read that is not complete cannot give it: such a read is refused before the list, with `partial chain read at block B: ...`.
   - The number after `the queue of` is `"queue"` of `status`: the seats that wait. It is not the size of the subnet.
   - The first `grep` line prints the number of `remove N seats`. The second prints `0`.
   - The seats in the list are really gone from the chain. Check a few UIDs in a chain explorer: the UID now has another hotkey or a later registration block.
   - There is no `WARNING` line (the second `grep` line prints `0`). The line `WARNING: not one of the N seats is on this chain. A read of another network or subnet looks like this` comes when not one seat of the queue is on the chain. It is a warning and not a refusal. With it, check `SN120_V2_CHAIN_NETWORK` first.
2. **Stop the loop** (step 10.4 of "GPU box"), as for `end-round`. Then `pgrep -af "sn120-v2-run-loop|sn120_v2 run"` must print nothing. **With `--confirm-remove`, the command takes over the journal before it reads the chain, also when it then refuses. So the loop is stopped before each try with `--confirm-remove`.**
3. **Apply, with the count of step 1.**

   ```
   python -m sn120_v2 force-sync --state-dir $STATE --stack sn120_v2.assembly:stack --confirm-remove 80
   ```

   Output (exit code 0): the same list again, then `applied at chain block 9500431: the queue has 259 seats` (256 seats, less the 80 that leave, plus the 83 that enter), then the same JSON summary that `status` prints (`"phase": "IDLE"`, `"queue": 259`). The list is long here too: add `> $CUT/force-sync-applied.txt; echo "exit=$?"` to the command, then read the file with `grep -n "applied at" $CUT/force-sync-applied.txt` and `tail -n 30 $CUT/force-sync-applied.txt`. A refusal line is not in the file: it comes on the screen.
4. **Start the loop again** (step 10.2 of "GPU box"). Check: `run.log` has no new `waiting: sync would remove` line, and a round opens.

The exit codes and the lines:

| Exit code | Line | Meaning | What to do |
|---|---|---|---|
| 0 | `nothing was changed. To apply exactly this, ...` | The list only. Nothing is written. | Check the list, then steps 2 and 3. |
| 0 | `applied at chain block B: the queue has N seats` | The sync is in the journal. | Step 4. |
| 2 | `refused: V2Error: the chain now removes X seats, not the confirmed N; nothing was changed. Read the list again` | The chain moved after step 1, or the count was typed wrong. | Step 1 again, then step 3 with the new count. |
| 2 | `refused: InfraError: two chain reads give different changes of the queue (remove X, add A at block B1 and remove Y, add C at block B2); nothing was changed. Run the command again` | A registration came between the two reads, or one read was wrong. | Run the same command again. **If it comes three times in a row, do not force:** the chain node is not steady. Send the lines to the coordinator. |
| 2 | `refused: InfraError: the chain stayed at block B for 60 s: there is no second read to compare with` | The chain node gives no new block. | Do not force. The chain endpoint must be repaired first. |
| 2 | `refused: ConfigError: this stack reads no chain, so there is no sync to apply` | `SN120_V2_CHAIN_NETWORK=none`. | Nothing to apply. |
| 2 | another `refused: InfraError: ...` from the chain read | The chain read failed or was not complete. | Run it again. |

Two refusals stay also with `--confirm-remove`: a read of another netuid, and a read with no registration.

What a forced sync costs a seat that it removes: nothing more than the normal sync. If such a seat comes back on a later read with the same UID and registration block, it enters again as a new seat at the place of its registration block. It loses the hold, the count of failures, and a front place that it earned in a round. Seats in an open round are never removed.

What is not proven:

- The command was run against the real chain on 14 Nov 2026 (reads only, netuid 120 on `finney`), with a local store and the CPU stack: the refusal of `run`, the list, a wrong count, and the right count for 333 of 999 seats. The real first line was `chain blocks 9501152 and 9501153 (1000 registrations) give the same changes to the queue of 999 seats:`. In all 4 runs the second read was at the next block.
- Not run on the real chain: the refusals `two chain reads give different changes` and `the chain stayed at block B`, the `WARNING` line, the real stack as a whole (only its chain read was real), and a queue with front seats, back seats, held seats or seats in an open round.
- The two reads are one block apart (about 12 s), at the newest block of the node. So the two-read guard finds a read that is wrong for one moment. It does not find a node that is wrong for longer.
- Both reads go through the same chain node. A node that is wrong in the same way at two blocks passes the two-read guard. Then only your look at the list stands, so do the checks of step 1.
- The block numbers and the counts in the examples above are made up, but they can be real: 1000 registrations, and a queue as long as the waiting seats of the import. The format of the lines is tested.

### A crash or a restart of `run`

The start, the look and the stop of the restart loop are step 10 of "GPU box". The head of `systemd/sn120-v2-run-loop.sh` lists every rule and setting. [SMOKE.md](SMOKE.md), "After a green test: the restart loop of `run`", has the same facts.

The rules for mainnet:

- **Only the loop starts `run`.** Never start `run` by hand.
- **Stop the loop before `set-params`, before `end-round` by hand, before `release-seat`, before `force-sync --confirm-remove`, and before a new commit.** These four commands take over the journal. Then start the loop again.
- **One box only.** Stop the loop on an old box (and `disable` its unit) before a new box starts. The loop of a second box ends the open round of the first one.
- **Run the `diff` line after each checkout, after each install line and on each new box, before any start of the loop.** `run` stops by itself on a box whose digests differ from the pinned ones, and no seat pays for it. But it does not see a stale `SN120_IMAGE_DIGEST`. "The digests of the box and the pinned digests" below has the line of `run`, the `diff` line, and what each one sees.

Before each start of `run` the loop does three things, in this order:

1. It stops every process that the last `run` left (vLLM servers, training jobs) and waits until no GPU holds more than 1,024 MiB. If a GPU stays in use for 120 seconds, the loop stops and says which GPUs.
2. It ends an open, undecided round with `end-round`: no decision, nobody is charged, no failure is counted, and the seats are first in the next round. If the box result of that round is already in the store, it does not end the round: `run` judges that result.
3. It starts `run`.

What a restart does to an open round:

| The round when `run` died | `run` alone, started by hand | The loop |
|---|---|---|
| In the training time (to minute 40) | All batches train again from zero. A batch of the first group that does not finish gets `no_decision`. When another batch of the round trained, that is **one counted failure** for that seat (3 hold the seat). When no batch trained, the lease fails and nothing is counted ("A round in which no training finished"). Nobody is charged. | `end-round`: no decision, nobody charged, no failure counted. The seats are first in the next round, which opens 5 minutes later at the earliest. |
| After the training time, no result stored | The round closes with no decision. Nobody charged, no failure counted. | The same, at once. |
| After the lease end, no result stored | The lease is failed, and the round closes with no decision: `no winner: the lease expired`. No charge, no counted failure. | The same effect, with another text: the loop's `end-round` runs first, so the summary says `no winner: ended by the operator`. |
| The box result is stored, not judged | `run` judges the stored result, also after the lease end. | The same: the loop skips `end-round`. |
| Decided (`"phase": "APPLY"`) | `run` finishes the promotion and the fold. The charges were fixed with the decision. | The same. |

So no restart can charge a seat. With the loop, no restart can count a failure. A restart by hand in the training time can. If you ever must restart `run` by hand with an open, undecided round (the loop itself does not work), use the order of the loop: stop `run`, free the GPUs, `end-round`, start.

**After a restart the page can show `no valid file` for every waiting seat. This is normal, and no seat pays for it.** A new process has not read the waiting files yet. What it publishes before its first read has `"files_checked_at": null` under `queue` in `mining.json`, and `"file_valid": false` for each waiting seat. The page shows `no valid file` for each of them, and `llms.txt` says `no check yet` for the last check of the files. The three cases:

- `end-round` closed a round (each start of the loop with an open, undecided round): for 5 to 7 minutes on the site. After a failed lease, `run` reads no file for 5 minutes (`apply_retry_s`, 300 seconds). Then come one pass of `run` (30 seconds) and the pull (each minute).
- A start of `run` in the 5 minutes after a round that ended with a failed lease: for the rest of that time.
- A start of `run` while the chain gives no answer: until the first pass that gets an answer.

A start in phase `IDLE` after a round that closed with a result does not show it: the first pass reads the files. No seat and no place in the queue is touched: the choice of the entries reads the files anew at each try to open a round, and never uses the published value.

Which exit codes of `run` the loop does not start again:

| Exit code of `run` | Meaning | The loop |
|---|---|---|
| 0 | Finished (`--rounds`, `--max-idle`) | stops, exit code 0 |
| 2 | Refused: a line `refused: <cause>` (parameters, state, a journal that another controller took, or a box whose digests differ from the pinned ones: `refused: ConfigError: this box differs from the pinned digests, ...`) | stops, exit code 2. Only `refused: InfraError` (the store or the chain did not answer) is tried again, without end, and is not counted. After the `ConfigError` line of the digests, nothing is counted for a seat: do what "After a fix" says (the digest block, `set-params` with `$CUT/params.json`, the loop). |
| 130, 143 | Stopped with SIGINT or SIGTERM | stops, exit code 0 |
| every other code | A crash (1 is a Python error, 137 is SIGKILL or out of memory) | starts `run` again after 30 seconds |

The loop counts the starts with no finished round between them. A round that `run` closed sets the count back to zero; a round that the loop ended with `end-round` does not. At 5 (`SN120_LOOP_MAX_STARTS`) the loop stops with exit code 3. A crash 30 minutes into every round stops it too. The count is in memory: a new start of the loop, or a reboot, sets it to zero. So the count does not stop a box that reboots in every round.

A store or a chain that does not answer never stops the loop. When `end-round` or `run` ends with `refused: InfraError`, the loop waits and tries again until they answer. The pause starts at 30 seconds, doubles with each try, and stays at 5 minutes (`SN120_LOOP_MAX_PAUSE_S`). Such a try is not in the count of starts, and it does not set that count back to zero. `cat $SN120_WORK_DIR/logs/run-loop.status` then shows `waiting for the store or the chain since <UTC time> (<N> tries with no answer); next try in <seconds> s; the last one ended with: ... refused: InfraError: <cause>`. With form B, `systemctl status sn120-v2-run` stays `active (running)` in that time. Read that `<cause>` once after each start of the loop. A cause on the box itself has the same word `InfraError` and does not go away: a wrong store token (for example `r2 ... failed: AccessDenied`), or a wrong grader package (`math grader needs ...`). Then stop the loop and fix it.

The loop stops by itself after 5 starts with no finished round between them, when `run` is refused, and when a GPU stays in use. Then nothing starts `run` again. The last lines of `$SN120_WORK_DIR/logs/run.log` say why and what to run. `cat $SN120_WORK_DIR/logs/run-loop.status` shows the same in one line, with the exit code of the loop: 0 stopped or finished, 2 refused, 3 gave up, 4 a GPU stays in use, 64 a wrong call. With form B, `systemctl status sn120-v2-run` then says `failed` (after exit code 0: `inactive`). Send those lines to the coordinator. While `run` does not run, no round opens; the head stays, and the weights writer keeps its last valid assessment.

A fault in the judging of a round stops `run` with a Python error. The round stays open, and the box result stays in the store. After the fix, `run` judges that result; no new round is needed. `end-round` leaves such a round alone and exits with 3. Only if the stored result itself is bad: `end-round --discard-result` ends the round with no decision and throws the result away. Nobody is charged.

### When the store or the chain fails while `run` runs

One pass of `run` is: read the chain, then the whole round. A failure of the store or the chain in a pass gives `waiting: <cause>`, then the `idle:` line, 30 seconds of sleep, and the next pass. `run` does not end, so the loop does nothing, and `run-loop.status` says `running`.

**The chain.** `run` reads it only at the start of a pass. Inside a round it is not read, so a chain fault in the middle of a round does nothing. Between rounds it gives `waiting:` at every pass, and no round opens. A decided round that waits for a new try of its promotion or its fold also waits for the chain. From `e9958463`, a chain read of `run` ends in 90 s at most, and after a failed read the next pass opens a new connection: a chain outage gives `waiting: chain read of ...` every pass and ends by itself when the chain answers again, with no restart of `run`. "When no round opens" has the lines and the numbers.

One chain line does not end by itself. After a stop of `run` of more than about a day, or on a queue of 8 to 11 seats, `run` can print `waiting: sync would remove N of M seats; refusing without force (possible bad chain read). If the chain is right, an operator applies it with the force-sync command` every 30 seconds, and it opens no round. The way out is `force-sync` ("When `run` refuses the chain sync").

**The store**, by the phase of the round. The store client tries each call again by itself, so a short fault is not seen at all.

| Phase | What `run` does | Cost |
|---|---|---|
| A round opens | `waiting:`. The next pass opens it again. | none |
| Training | Not `waiting:`, not a crash. The runner tries that batch 3 times at once. Then the batch is `no_decision` ("training failed 3 times"). `run` goes on with the other batches and the scoring. | From `2b7046a3` on: no counted failure and no charge; the seat is first in the next round. (A commit before it counted one failure for that seat when another batch of the round trained.) When no batch of the round trained: the lease fails, and nothing is counted ("A round in which no training finished"). |
| Scoring | 3 tries for a block of tasks. After the stop time the block is dropped and the round ends with what it has. Before it, the runner gives up: the lease fails and the round closes with no decision. | no charge, no counted failure; the next round opens 5 minutes later |
| Training or scoring, and the store does not answer the controller either | `waiting:`. The lease stays open. The next pass runs the whole round again under the same lease. | in the training time: a counted failure for a batch of the first group that does not finish, when another batch of the round trained. After it: none |
| The box result is written | As the row above: the result was only in memory, so the scores are lost. | none, but the round decides nothing |
| Judging (the result is stored) | `waiting:`. The next pass judges the stored result. | none |
| Promotion | Store fully down: `waiting:`, and no try is counted. The journal works but the copy fails: a new try every 5 minutes; after 12 tries the round is `abandoned`. | `abandoned`: no new head, nobody charged |
| Fold | `fold failed`, a new try every 5 minutes, with no limit. No new round opens before the fold is done. | none |
| The close of the round | `waiting:`. The next pass closes it. | none |
| Right after the close | `run` ends with `refused: InfraError`, exit code 2. This is the one place where a store fault ends a running `run`. The round is closed and safe. | none; the loop starts `run` again, and the start is not counted |

So, from `2b7046a3` on, a store fault that makes a training job fail costs a seat nothing. One row still costs a counted failure, and only in a round in which another batch trained: the pass that runs the whole round again in the training time, when a batch of the first group then does not finish. When no batch of the round trained, the lease fails and nothing is counted. Send the coordinator the round summary of such a round. "After a fault of the box in training" below says how to undo the counts: `release-seat --all`.

### After a fault of the box in training

This part is for a fault of the box that is not a digest difference: a GPU fault, or a store fault during the training (the row "Training" of the table above). The guard of `run` does not see such a fault. What is counted for a seat, from commit `2b7046a3` on:

- **A training job that failed is never counted**: out of memory, a crash of the job, any other error of the training. The entry is `no_decision` with a `reason` that starts with `training failed`. The seat is first in the next round.
- **One failure is counted**, and only when another batch of the round trained: a batch that started at the lease, whose first try ran to the end of the training time and did not finish. Its `reason` starts with `this batch had the whole training time of the round`.
- From commit `e35a09c4` on: when a try of a batch failed and a later try is stopped at the end of the training time, the batch is `not_trained`, and nothing is counted. With an earlier commit this case is counted.
- A commit before `2b7046a3` counted each failed training. A count that such a commit wrote stays in the state after the checkout: `release-seat` takes it away.

A seat with 3 counted failures (`max_infra_attempts`) is held: it takes no turn. A count never goes back by itself. Nobody is charged. When the fault stops every training of the round, nothing is counted: see "A round in which no training finished" below.

`release-seat` undoes both: the hold and the counts. It is there from `c87b81ea` (check 1 of "Before the mainnet bootstrap").

1. Stop the loop (step 10.4 of "GPU box"). `release-seat` takes over the journal, as `set-params` does. Under a running loop, `run` ends at its next write with `refused: FencedError`, exit code 2, and the loop stops.
2. Repair the box.
3. Look at the seats:

   ```
   python -m sn120_v2 status --state-dir $STATE --stack sn120_v2.assembly:stack
   ```

   `"held_seats"` is the list of the UIDs of the held seats. `"seats_with_counted_failures"` is the number of seats that have a counted failure and are not held. With `[]` and `0` there is nothing to release: go to step 5.
4. Release the seats. One of `--all` and `--uid` is necessary.

   ```
   python -m sn120_v2 release-seat --state-dir $STATE --stack sn120_v2.assembly:stack --all       # every seat that is held or has a count
   python -m sn120_v2 release-seat --state-dir $STATE --stack sn120_v2.assembly:stack --uid 12    # one UID; give --uid again for more
   ```

   - It ends with exit code 0 and prints the released seats: `{"released": [{"uid": ..., "registration_block": ..., "hotkey": ..., "was_held": true, "counted_failures": 3}, ...]}`. Put the output with the run records.
   - A released seat is not held, its count is 0, and its place in the queue does not change. `--all` also sets back the counts of the seats that are not held.
   - It is refused with exit code 2 when it finds no such seat, and then it releases nothing: `refused: V2Error: no seat is held or has a counted failure; nothing was released`, or `refused: V2Error: no seat of UID 12 is held or has a counted failure; nothing was released`. With more than one `--uid`, one UID with no hold and no count refuses the whole command.
   - **Do not release a seat whose own file made the training fail.** It is held again after 3 more rounds. If you do not know which seats the fault of the box stopped, send the `status` output and the round summaries to the coordinator before `--all`.
5. Start the loop (step 10.2 of "GPU box"). After `--all`, `status` shows `"held_seats": []` and `"seats_with_counted_failures": 0`.

### A round in which no training finished

The rule, from `68b8abc5` (check 1 of "Before the mainnet bootstrap" tests it with `f986ccb1`): when no batch of a round trained, and at least one batch ended as `no_decision`, `run` takes the fault as a fault of the box. The round ends as a failed lease. There is no decision, nobody is charged, and no failure is counted for a seat. The seats are first again, and the next round opens `apply_retry_s` (5 minutes) later. `run` goes on, and the loop does not stop.

- "Trained" means that the entry is `scored` or `not_scored`. A skipped entry or a `not_trained` entry in the same round changes nothing.
- Both kinds of `no_decision` are in the rule: "training failed 3 times", and a batch that had the whole training time of the round and did not finish (a slow or hung box).
- A round in which another batch trained: from `2b7046a3` on, a batch whose training job failed is not counted either. Only a batch that had the whole training time and did not finish is one counted failure for its seat ("After a fault of the box in training").

What you see in `run.log`, and with `python -m sn120_v2 round --state-dir $STATE --stack sn120_v2.assembly:stack`:

```
SN120 v2 round r00000N: NO_DECISION
no winner: InfraError: no training finished in this round: 32 of 32 batches failed. The first: training failed 3 times: <the cause of the first batch>
head: <12 characters> (unchanged)
```

`<the cause of the first batch>` is the safe form of the error text ("Error text in the public round file" below).

Between such rounds `run` prints `idle: N seats wait, phase IDLE`. `status` shows `"no_decision"` at the end of `"last_outcomes"`, and `"held_seats"` does not change.

Every round that closes with no box result names its cause in the same place:

- `no winner: ended by the operator` after `end-round`, by hand or by the loop.
- `no winner: LeaseError: the lease has expired; a late result is never used` when a `run` that is alive brings its result after the lease end.
- `no winner: the lease expired` only when `run` is started by hand after the lease end, with no result in the store. Through the loop this text never comes: the loop's `end-round` runs first, and the text is the first one.
- `no winner: <error class>: <text>` after another fault of the box.

The first three have the same effect: the round closes with no decision, nobody is charged, no failure is counted, and the seats are first in the next round.

**The cost of this rule.** `run` cannot tell a broken box from bad files.

1. A seat that is alone in its round, with a file that makes the training fail, is never held. Its round repeats every 5 minutes, with 3 training tries each, until the miner changes the file or another batch trains in the same round.
2. The seats of a failed lease are first again. So `batch_cap` (32) such seats in front fill every round, and no seat behind them gets a turn. No command holds such a seat or moves it back.
3. Nothing stops the loop for it. **The sign is `"no_decision"` again and again in `"last_outcomes"` of `status`**, and the same `no winner:` line every 5 minutes.

What to do when you see that sign:

- Read the cause in the `no winner:` line. It is the safe form; the full text is in `run.log` ("Error text in the public round file" below). If the box is the cause (a GPU fault, a full disk, the store), stop the loop, repair the box, and start the loop. No seat was held and nothing was counted, so `release-seat` is not needed for these rounds.
- If the box is good and the files of the seats in front are the cause, there is no command for it today. At the launch values, no `set-params` lets other seats into the round. `batch_cap` cannot go above `max_loras_per_box`, and both are 32: a file with a higher `batch_cap` is refused with `refused: ConfigError: batch_cap must be between 1 and max_loras_per_box` (exit code 2, the parameters do not change). Do not raise `max_loras_per_box` to get around it: it is the number of LoRAs in one scoring server, and no test has run more than 32. Send the `no winner:` lines and the `status` output to the coordinator: the fix is a code change.

### Error text in the public round file

The round file `public/rounds/<id>/round.json` is public. It holds text from errors in three places: the `reason` of the round, the `reason` of each row, and the `note`. From `f986ccb1` (check 1 of "Before the mainnet bootstrap"), `run` makes each such text safe before it writes the file. The error class and the message stay. A marker stands in the place of each private part:

| Marker | What it stands for |
|---|---|
| `[removed-secret]` | The value of an environment variable of `run` whose name holds `KEY`, `SECRET`, `TOKEN`, `PASSWORD`, `PASSWD`, `CREDENTIAL`, `ENDPOINT` or `ACCOUNT`, when the value has 8 characters or more. Also the value after a name such as `token=` or `Authorization: Bearer`. |
| `[removed-query]` | The query string of a URL (the part after `?`). A signed link holds its signature there. |
| `[removed-address]` | An IPv4 or IPv6 address. The port stays. |
| `[removed-login]` | The login of a URL (the part before `@`). |

A text of more than 600 characters is cut. It then ends with ` [cut: N more characters]`.

The `no winner:` line of the round summary and the `round` command read the public file. So they show the safe form:

```
no winner: InfraError: evaluation failed 3 times: http://[removed-address]:8000/v1/completions failed: ...
```

**Where the full text is.** In three places, and none of them is public:

- `run.log`: a line `round <id>: the public round file holds a changed form of this text. The full text: <text>`. The line comes only when the public form differs from the full text. To find it: `grep "holds a changed form of this text" $SN120_WORK_DIR/logs/run.log | tail -n 5`.
- The journal: the `reason` of the records `lease_failed` and `round_closed`.
- The store: `private/rounds/<id>/<lease>/result.json`.

**If a `[removed-secret]` marker stands for a real credential** (the line in `run.log` shows the value): the public file is safe. But `run.log`, the journal and the sealed result hold the value. Keep them private: do not paste such a line into a public place, and tell the coordinator which credential it is, not its value. Change that credential after the round.

**The limits.** A secret that `run` read from a file, and not from an environment variable, is found only by the query rule or the name rule. Host names and paths on the box stay in the text. A round file that was written before `f986ccb1` is not written again.

### The digests of the box and the pinned digests

The pinned digests are the two digests of the parameters in force: the code digest and the image digest. The last bootstrap or `set-params` pinned them. A checkout can change the code digest of the box (a file under `sn120_v2/gpu/`). An install line can change the image digest of the box (a package). A new box can have other digests than the old one.

**`run` stops by itself on a box whose digests differ from the pinned ones.** This is the guard of `e0faff2d` (check 1 of "Before the mainnet bootstrap"). `run` compares the box with the pinned digests at three points: before it opens a round, before it runs an open round again, and before a fold. On a difference it starts none of them, counts no failure for a seat, charges nobody, and ends with exit code 2 and one line. The loop does not start a refused `run` again: it stops with exit code 2 (with form B, `systemctl status sn120-v2-run` says `failed`).

A box result that is stored and not judged is judged first, on any box. The judging reads the digests of the round, not those of the box.

The line:

```
refused: ConfigError: this box differs from the pinned digests, so <what did not start>. The difference: <difference>. Nothing is counted for a seat. Next: the digest block, then set-params with the whole parameter file in which only the two digests are new (set-params replaces every value: a key that the file does not hold goes back to its default), then start again
```

- `<what did not start>` is one of `no round was opened`, `the fold was not started`, `the open round was not run`. With the third one, the line says `Next: end-round (the round pins the old digests), the digest block, ...`.
- `<difference>` is `image digest: pinned <value>, this box has <value> (the value of SN120_IMAGE_DIGEST)`, or `code digest: pinned <value>, this box has <value>`, or both with `; ` between them. When more than one image digest is pinned, the line shows `<value> or <value>`.
- No other command shows the pinned digests.
- The words after `Next:` above are in the code from `68b8abc5`. An earlier commit ends the line with `Next: the digest block, then set-params with the two digests of this box, then start again`. The meaning is the same: not a file that holds only the two digests.
- **The file for `set-params` after this line is `$CUT/params.json` as the digest block of "After a fix" wrote it, never a file written by hand.** That block takes the pinned file and changes the two digests only. `set-params` replaces every value: a file without `"squat_min_wait_s": 31536000` switches squat-trim on ("The value `squat_min_wait_s`").
- What to do after the line: "After a fix" for a fix, "A new GPU box" for a new box.
- To see the line after the loop stopped: `grep '^refused: ' $SN120_WORK_DIR/logs/run.log | tail -n 1`. `run-loop.status` says `stopped: run refused (exit code 2 of the loop)`.

Before `e0faff2d`, `run` did not stop on such a box: every training was refused at once, each round closed in seconds, and every seat in it got one counted failure. That is history, and it is the reason why check 1 tests `e0faff2d`.

The guard has three limits:

1. The image digest of the box is the value of `SN120_IMAGE_DIGEST`. `run` does not compute the digest of the installed packages again. So it does not see a stale value: after an install line with no digest block, the value is still that of the old packages, nothing is refused, and the rounds run with packages that the pinned image digest does not name. The `diff` line below sees it.
2. The code digest is computed from the files at each check. A checkout under a running `run` is refused at the next round or fold, not in the middle of a round. So stop the loop before a checkout, as before.
3. The guard does not see a fault of the box that is not a digest difference (a GPU fault, a store fault during the training). Such a fault still gives counted failures when another batch of the same round trained: see "After a fault of the box in training". When no batch of the round trained, nothing is counted ("A round in which no training finished").

**Run the `diff` line after each checkout, after each install line and on each new box, before any start of the loop:**

```
python -m sn120_v2.gpu.plan | diff - $CUT/plan.json; echo "exit=$?"
```

- `exit=0` and no other output: the digests of this box are those of `$CUT/plan.json`, the output of the last digest block. The value of `SN120_IMAGE_DIGEST` comes from that file, so it is not stale. The loop can start.
- A `diff` output and `exit=1`: the checkout or the install line changed the digests of the box. The digest block and `set-params` come before the start ("After a fix"), also when the message for the fix did not say so.
- `$CUT/plan.json` holds the pinned digests only if a bootstrap or a `set-params` followed that block. A new box compares with the `plan.json` of the old box ("A new GPU box").

Which of the two you read, and when:

| | The `diff` line | The `refused:` line of `run` |
|---|---|---|
| When | You run it by hand before a start of the loop: after a checkout, after an install line, on a new box. | The loop stopped with exit code 2. The line is in `$SN120_WORK_DIR/logs/run.log`. |
| What it compares | The digests of the box as they are now (the files and the installed packages) with `$CUT/plan.json`. | The pinned digests with the code digest of the files and with the value of `SN120_IMAGE_DIGEST`. |
| What it does not see | The pinned digests. | A stale `SN120_IMAGE_DIGEST`. |
| What it is for | It says before the start that the digest block is needed. | It shows the pinned digests and those of this box. It protects the seats when you did not see the difference before the start. |

A commit that changes `v2/pyproject.toml` does not need the install line and the digest block for that reason alone. The message for the commit says if the install line is needed. Example: `4e23a275` adds `jinja2` to the `miner` line. A box with the `gpu` extra has that package already, so this commit is a checkout only. The `diff` line shows it for this box.

### The value `squat_min_wait_s`

The mainnet `params.json` holds `"squat_min_wait_s": 31536000` (check 5 of "Before the mainnet bootstrap"). With it, squat-trim moves no seat whose registration is younger than 730 days. A seat from v1 with no valid file keeps its place and is not charged.

**Each later file for `set-params` must hold the same key.** `set-params` does not add a file to the pinned parameters. It replaces them, and a key that the file does not hold goes back to its default (86400). So a file with only the two new digests switches squat-trim on. Run the check line of check 5 on each file before `set-params`: its first number must be `31536000`. In phase `APPLY`, `set-params` refuses such a file, because it differs from the pinned one in more than the two digests. In phase `IDLE` nothing refuses it.

**Read `changed` after each `set-params`.** From `6baf0d74` (check 1 of "Before the mainnet bootstrap" tests it with `f986ccb1`), `set-params` prints a third key, `changed`: each parameter whose value differs from the pinned one, with the value `before` and `after`. A value of the recipe has the name `recipe.<name>`. The output is still one line of JSON, with the two hashes first. This is the output for a file that lacks `squat_min_wait_s` and has a new code digest:

```
{"params_hash_before": "4a17...c253", "params_hash": "2658...4d49", "changed": {"squat_min_wait_s": {"before": 31536000, "after": 86400}, "recipe.code_digest": {"before": "", "after": "sha256:abab...abab"}}}
```

- The same file a second time prints `"changed": {}`, and the two hashes are the same.
- After the digest block of "After a fix", `changed` holds `recipe.code_digest`, `recipe.image_digests`, or both, and no other key.
- A line `"squat_min_wait_s": {"before": 31536000, "after": 86400}` means that the file lacks the key. **Do not start the loop.** Give `set-params` the right file: the digest block of "After a fix" writes it. `changed` then shows the reverse, `{"before": 86400, "after": 31536000}`. `params_hash` is the hash that the check line prints for the right file: the parameters are hashed by content, so a file equal to the one that was pinned before gives the old hash again.
- Any other key that you did not want to change: the same. Do not start the loop, and give `set-params` the right file.
- While the loop is stopped, no round uses the wrong value, so nothing is lost on the box. `set-params` writes two records to the journal (the takeover and one `params_set`). It moves no seat and charges nobody. Squat-trim reads the value only when `run` picks the entries of a round.

**One limit: the weights writer does not wait for the loop.** The weights writer on AffineValidator reads five values from the pinned parameters at minute 05 of each hour: `kings_window`, `burn_uid`, `netuid`, `weights_cadence_s` and `weights_rate_limit_blocks`. If `changed` holds one of these five by mistake, put it right before the next minute 05.

Do not run `publish` by hand between the wrong `set-params` and the right one. It would put the wrong values into the public snapshot (`llms.txt`, `mining.json`), and they would stay there until the next publish.

At a commit before `6baf0d74`, `set-params` prints only the two hashes. Then the check line on the file before `set-params` is the only check.

How to see what is pinned, in two ways:

- The check line of check 5 prints the hash of a file's parameters. `"params_hash"` of `status` is the hash of the pinned ones. The same hash means that this file is pinned.
- `llms.txt` lists every parameter: `grep squat_min_wait_s` on that file prints the row of the parameter table, with the name and the number (`31536000` for the mainnet file). From `6233de06` it also prints one line of text that names the parameter.

**To switch squat-trim on later** is Jacob's decision. It is a `set-params` between two rounds (phase `IDLE`, the loop stopped) with a file that holds the wanted number, or no such key for the default. Know this before you do it: **at that moment every old seat with no valid file moves to the back at once.** More exactly: at the next try to open a round, if an entry was charged in the last 24 hours, each waiting seat of the normal part of the queue that has no valid file and a registration older than 2 times the new number moves to the back (older still, when many seats wait ahead of it and few entries are charged in a day). Front seats do not move. A moved seat stays at the back, also when a valid file comes later. No command moves it forward again.

### After a fix: a new commit on the GPU box

This is "After a fix" of [SMOKE.md](SMOKE.md). Three things differ on mainnet: there is no `smoke params` command, so the digest block below puts the two new digests into `$CUT/params.json`; that block starts with the lines of check 3 of "Before the mainnet bootstrap"; and the loop is stopped first and started last, where SMOKE.md says `run`.

A fix is a new commit on `v2`. The state continues at the new commit: there is no new bootstrap. Never run `bootstrap` again. The message for the fix says if it needs the install line and if it needs the digest block. The commit of a fix always holds `9b830458`: there is no way back below it (check 1 of "Before the mainnet bootstrap").

**Run the `diff` line after the checkout and the install line, before any start of the loop.** `run` refuses a box whose digests differ from the pinned ones, and no seat pays for it. But `run` does not see a stale `SN120_IMAGE_DIGEST` ("The digests of the box and the pinned digests").

Stop the loop (step 10.4 of "GPU box"), not `run` alone: a `kill -9` of `run` makes the loop start it again. A stop in the middle of a round costs that round: the loop ends it with no decision at its next start. Nobody is charged, no failure is counted, and the seats are first in the next round. Then look at the state:

```
python -m sn120_v2 status --state-dir $STATE          # "phase": "IDLE", "RUNNING" or "APPLY"; write down "params_hash"
```

Then:

```
git fetch origin && git checkout <the commit of the fix>
uv pip install -e ".[dev,gpu,r2,chain,envs,miner]"    # only when the fix needs the install line
python -m pytest
python -m sn120_v2.gpu.plan | diff - $CUT/plan.json; echo "exit=$?"
```

**The `diff` line prints only `exit=0`: the fix needs no digest block.** Start the loop again. In phase `RUNNING` the round must end first (stop `run`, free the GPUs, `end-round`, start): a start of `run` in the training time of an open round can count a failure for a seat. The loop does this order by itself, so there is nothing more to do.

A fix for a fault in the judging, with no digest block: the round is open and its box result is in the store ("A crash or a restart of `run`"). Start the loop: `run` judges the stored result.

**The `diff` line prints an output: the fix needs the digest block** (a package changed, or a file under `sn120_v2/gpu/` changed). The block is the same in every phase. It changes only files under `$CUT` and one variable of the shell:

```
python -c "import os; from sn120_v2.params import load_params; p = load_params(os.environ['CUT'] + '/params.json'); print(p.squat_min_wait_s, p.params_hash)"     # the pinned file: 31536000 and the "params_hash" of status
python -m sn120_v2.gpu.plan | tee $CUT/plan.json
export SN120_IMAGE_DIGEST=$(python -c "import json,os; print(json.load(open(os.environ['CUT']+'/plan.json'))['image_digest'])")
cp $CUT/params.json $CUT/params-pinned.json
python -c "import json,os; c=os.environ['CUT']; plan=json.load(open(c+'/plan.json')); p=json.load(open(c+'/params-pinned.json')); p['recipe']['image_digests']=[plan['image_digest']]; p['recipe']['code_digest']=plan['code_digest']; json.dump(p, open(c+'/params.json','w'))"
python -c "import os; from sn120_v2.params import load_params; p = load_params(os.environ['CUT'] + '/params.json'); print(p.squat_min_wait_s, p.params_hash)"     # the new file: 31536000 and a new hash
```

- The first line must print `31536000` and the `"params_hash"` that `status` showed. Then `$CUT/params.json` is the pinned file. With another hash, stop: the file is not the pinned one. Tell the coordinator.
- The fifth line writes `$CUT/params.json` again: the pinned file with the two digests of `$CUT/plan.json`, and no other change. **This is the file for `set-params`. Never write that file by hand**, and never give `set-params` a file that holds only the two digests: `set-params` replaces every value, and a key that the file does not hold goes back to its default ("The value `squat_min_wait_s`").
- The last line must print `31536000` and a hash that differs from the first one.
- With form B, write `$CUT/run.env` again before the start (the first two lines of the block of form B, step 10.2 of "GPU box"): the digest block can change `SN120_IMAGE_DIGEST`, and the unit reads it from that file.

Do not start the loop yet. What comes next depends on the phase:

- `IDLE`: `set-params` with `$CUT/params.json`, then start the loop.

  ```
  python -m sn120_v2 set-params --state-dir $STATE --stack sn120_v2.assembly:stack --params $CUT/params.json
  ```

  It prints `{"params_hash_before": "...", "params_hash": "...", "changed": {...}}`. `params_hash` must be the hash of the last line of the digest block. **Read `changed`:** it must hold `recipe.code_digest`, `recipe.image_digests`, or both, and no other key. With another key, do not start the loop: see "Read `changed` after each `set-params`" in "The value `squat_min_wait_s`". Put the new `$CUT/params.json` and `$CUT/plan.json` with the run records (step 11 of "GPU box"): a new box needs them.
- `RUNNING` (the round was not decided): the open round pins the old digests, so this box can no longer finish it. End it by hand first, with the loop stopped. `set-params` before it is refused: `refused: V2Error: a round is open; set the parameters when it has closed (a decided round takes new digests only)`.

  ```
  python -m sn120_v2 end-round --state-dir $STATE --stack sn120_v2.assembly:stack
  ```

  Exit code 0: the round is ended. Nobody is charged, no failure is counted for a seat, and its seats are first in the next round. Then do the `IDLE` line. Exit code 3: a box result of the round is in the store, and the round stays open. Then do the part below; it costs nobody anything.
- `APPLY` (the round is decided; the promotion or the fold is left): do the `IDLE` line. In this phase `set-params` takes only a file that differs from the pinned one in the two digests, and the file of the digest block is such a file. A file with another change is refused with the `V2Error` line above; make that change after the round is closed. `run` then finishes the fold with the new code. The sign that the fold is done: `run` prints `idle: N seats wait, phase IDLE`, or `status` in a second shell shows `"phase": "IDLE"`. `run` prints no round summary for it. If valid files wait, `run` does not print the line: it starts the next round at once.

If the loop starts before `set-params`, nothing is lost. The loop ends an open round with no stored result at no cost. Then `run` is refused with the line of "The digests of the box and the pinned digests" (`so no round was opened`, or `so the fold was not started`), and the loop stops with exit code 2. Then do the `IDLE` line.

`end-round` is also the command for a round that cannot finish for any other cause, for example when the box is gone. Stop the loop first. It is refused (exit code 2, `no round is open and undecided`) when no round is open, and when the round is already decided.

After any failed lease, the next round opens 5 minutes later at the earliest (`apply_retry_s` 300).

#### A checkout only: a running box goes to a later commit

This is the short form of the part above, for a commit that changes no digest. The message for the commit says "a checkout only". `0c29cb6c` is such a commit for a box at `1d7d0628` or later (check 1 of "Before the mainnet bootstrap"). `run` loads the code only at a start, so a checkout changes nothing in a `run` process that runs: the stop and the start are needed.

A commit with the command `front-seats` goes to AffineValidator first: see "Seats to the front by hand: `front-seats`".

1. **Before the stop: no round is open.** The cheapest moment for the stop is right after a round closes. In a second shell with the same exports, `python -m sn120_v2 status --state-dir $STATE` shows `"phase": "IDLE"`. Anybody can read the same value: `"phase"` in `https://affine.io/mining.json`, once the site shows v2's file.
   - `RUNNING` or `APPLY`: wait for `IDLE`. The promotion of a winner comes at most 100 minutes after the lease. The fold comes after it and has no time limit; during the fold `status` shows `APPLY` and the site still shows `RUNNING` ("The time limit of a round"). `run` prints the round summary at the close.
   - When valid files wait, `run` opens the next round at once, and no look shows `IDLE`. Then stop the loop right after a round summary in `run.log`: the round that is lost is the one that opened a moment ago. The table of step 2 says what a stop in a round costs: that round, and nothing for a seat.
2. **Stop the loop** (step 10.4 of "GPU box"). Then run the `status` line again. Go on in each case: the start of step 4 does the right thing by itself.

   | `status` after the stop | What the start does | The cost |
   |---|---|---|
   | `IDLE` | Nothing is left from a round. | None. |
   | `RUNNING`, no box result in the store | The loop's `end-round` ends the round with no decision. `run.log`: `run-loop: end-round: the open round is ended with no decision. Nobody is charged and no failure is counted.` | That round: its training and its scores are discarded. Nobody is charged, no failure is counted, and its seats are first in the next round. The next round opens 5 minutes later at the earliest (`apply_retry_s` 300), with a new draw of tasks. |
   | `RUNNING`, the box result is in the store, not judged | The round stays open: `end-round` ends nothing (exit code 3). `run` judges the result in its first pass. `run.log`: `run-loop: end-round: a box result of the open round is in the store. The round stays open, and run judges that result.` | None. |
   | `APPLY`, the round is decided, the winner not promoted | `run` promotes the winner, folds and closes the round. | None. |
   | `APPLY`, the winner promoted, the fold not done | `run` folds and closes the round. | None. |
3. **The checkout, then the `diff` line.**

   ```
   git fetch origin && git checkout <the commit>
   git rev-parse HEAD
   python -m sn120_v2.gpu.plan | diff - $CUT/plan.json; echo "exit=$?"
   ```

   The second line prints the commit. The `diff` line prints only `exit=0`. With a difference in its output and `exit=1`, the commit is not a checkout only: do not start the loop, and do the digest block of the part above.

   **`diff: .../plan.json: No such file or directory` and `exit=2`: the `diff` line checked nothing.** `$CUT/plan.json` is not there: the state was not made with the steps of "GPU box", or `$CUT` is not set in this shell (`echo $CUT`). Do this check in its place. It needs v2's `mining.json` on the site.

   ```
   python -m sn120_v2.gpu.plan
   echo $SN120_IMAGE_DIGEST                              # with form B: grep '^SN120_IMAGE_DIGEST=' $CUT/run.env
   curl -fsS https://affine.io/mining.json | python3 -c "import json,sys; r = json.load(sys.stdin)['params']['recipe']; print(r['code_digest'], r['image_digests'])"
   ```

   - The first line prints `{"code_digest": "...", "image_digest": "..."}`: the two digests of this box at this commit.
   - The third line prints the two pinned values. On 14 Nov 2026 they were `4366d82d3a2d68856b2da0876bdfbe0ce31c00ff661cddd0a08e8033d93a410f` (code) and a list with `9a4b16ff282b21baf44350e54c852ed59586b9efb6fe09e5cd2f1e288f4c108a` (image).
   - The check passes when the `code_digest` of the first line is the code digest of the third line, its `image_digest` is in the list of the third line, and the second line prints that same image digest.
   - Then save the file, so that the `diff` line works the next time: `python -m sn120_v2.gpu.plan > $CUT/plan.json`.
   - When one value differs: do not start the loop, and tell the coordinator.
4. **Start the loop** (step 10.2 of "GPU box": the line of form A again, or `sudo systemctl start sn120-v2-run`). Read the status line once (step 10.3 of "GPU box").

   **Form A in a new shell: check the variables before the start.** With form A the loop takes its variables from the shell that starts it. With form B the unit reads them from `$CUT/run.env`, which the checkout does not change.

   ```
   env | grep '^SN120_' | sed 's/=.*//' | sort           # the names only, no value
   echo $SN120_LINKS_URL
   ```

   - The first line prints the nine names of the check of form B (step 10.2 of "GPU box"). `SN120_R2_PREFIX` is not there.
   - The second line prints `https://affine.io/ff/links`. **`SN120_LINKS_URL` is the one variable that no command demands.** Without it the loop starts, and the first publish of the new `run` writes `llms.txt` and `mining.json` with no address for the upload links (`Where to get the link: not published yet`). Then a miner cannot find its link.
   - A missing store variable, `SN120_PUBLIC_URL` or `SN120_V2_CHAIN_NETWORK`: the loop's `end-round` is refused with `refused: ConfigError: environment variable <name> is not set; refusing to guess a default`. The loop stops, and `run-loop.status` says `stopped: end-round refused (exit code 2 of the loop)`. An open round stays open, and nothing is counted. Export the variable and start the loop again.
   - A missing `SN120_IMAGE_DIGEST`: only `run` reads it. So the loop's `end-round` runs first (the table of step 2), then `run` is refused with the same `ConfigError` line, and the loop stops: `stopped: run refused (exit code 2 of the loop)`. Export it (check 3 of "Before the mainnet bootstrap") and start the loop again.
   - A missing `SN120_WORK_DIR`: the loop does not start (exit code 64, `SN120_WORK_DIR is not set`). The line of form A sends this message away, so the sign is that `run-loop.status` has no new line.
5. **What you see.** `run` prints no line for a publish. A new `run` process writes `mining.json` and `llms.txt` into the store in its first pass:
   - No round opens in that pass: `run.log` shows `idle: N seats wait, phase IDLE`, and `run` writes the two files right after this line. At a later pass with no round, `run` writes them only when the queue or the result of a file check changed.
   - A round opens in that pass: `run` writes the two files at the open of the round, and `run.log` shows no `idle:` line.
   - **A round that the start judges or finishes gets no summary in `run.log`** (the last three rows of the table of step 2). The log goes straight to `idle: N seats wait, phase IDLE`, or to the next round. Read the result of that round with `python -m sn120_v2 round --state-dir $STATE --stack sn120_v2.assembly:stack` ("What a closed round looks like").
   - The sign in the file: line 3 of `llms.txt`, `Generated <time> UTC from snapshot`, shows the time of that pass. `generated_at` in `mining.json` is the same second.
   - `idle: ..., last problem: publish failed: <cause>`: the store did not take the files in the pass before. The next pass with no round writes them again, 30 seconds later. The text `last problem:` stays in the later lines, so the sign of the write is the `Generated` line, not the log.
   - The way to the site: after step 12 of "AffineValidator", the pull copies the two files within about a minute (its timer starts a run at second 30 of each minute). Before step 12 nothing copies them: run the command of step 8.4 of "AffineValidator" again.
6. **The check.** Two lines on the public file:

   ```
   curl -s https://affine.io/llms.txt | grep -c "A round with no result"
   curl -fsS https://affine.io/llms.txt | grep 'Where to get the link'
   ```

   The second line still names `https://affine.io/ff/links/<hotkey>.json`. If it says `not published yet`, the new `run` has no `SN120_LINKS_URL`: do the repair of step 13 of "AffineValidator".

   The first line prints `1` (step 8.2 of "AffineValidator"). If it prints `0` two minutes after the start, read line 3 of the public file (`curl -s https://affine.io/llms.txt | sed -n 3p`):
   - The time is older than the start: the write or the copy did not happen. Read the last lines of `run.log`, and look at the last run of the pull ("Public pull and site").
   - The time is that of the start or later: the `run` process does not run the new commit. Run `git rev-parse HEAD` on the GPU box, and read the `run-loop:` lines of `run.log`: there must be a new `loop started` line after the checkout.

#### A stored result that is not judged, and a box with other digests

This is the case where `end-round` exits with 3 and the box has other digests than the pinned ones: a fix with the digest block, or a new box. One order fits both. It costs nobody anything: no result is thrown away, nobody is charged, and no failure is counted. It needs the guard of `e0faff2d`.

Before it: the loop is stopped (on a new box: not started yet). The checkout, the install line when the fix needs it, and the digest block above are done. No `set-params` yet: in phase `RUNNING` it is refused.

1. `python -m sn120_v2 status --state-dir $STATE --stack sn120_v2.assembly:stack` shows `"phase": "RUNNING"`.
2. `python -m sn120_v2 end-round --state-dir $STATE --stack sn120_v2.assembly:stack` prints `kept: round <id> has a box result in the store (<lease>); run judges it. Nothing was ended. --discard-result ends the round without it.` and ends with exit code 3. The loop does the same at each start. On a new box this command also brings the journal copy of the store to the empty state directory.
3. Start the loop (step 10.2 of "GPU box"). `run` judges the stored result. Then `run` is refused with the line of "The digests of the box and the pinned digests" and exit code 2, and the loop stops with exit code 2 (with form B, the unit is `failed`). **This stop is a step of the order, not a fault.** `run` prints no round summary at the refusal. There are two results:
   - **A winner.** The winner is promoted (the head pointer and the kings). The line says `so the fold was not started`. `status` shows `"phase": "APPLY"`.
   - **No winner.** The round is closed. The line says `so no round was opened`. `status` shows `"phase": "IDLE"`. `python -m sn120_v2 round --state-dir $STATE --stack sn120_v2.assembly:stack` prints the summary of that round.
4. `set-params` with `$CUT/params.json` as the digest block wrote it:

   ```
   python -m sn120_v2 set-params --state-dir $STATE --stack sn120_v2.assembly:stack --params $CUT/params.json
   ```

   It prints `{"params_hash_before": "...", "params_hash": "...", "changed": {...}}`. **Read `changed`:** it must hold `recipe.code_digest`, `recipe.image_digests`, or both, and no other key. With another key, do not start the loop: see "Read `changed` after each `set-params`" in "The value `squat_min_wait_s`". In `APPLY` the file must differ from the pinned parameters in the two digests only (`recipe.image_digests`, `recipe.code_digest`); the file of the digest block is such a file. In `IDLE` any file is taken, so the hash check of the digest block and `changed` are the two checks there.
5. Start the loop. After a winner, `run` finishes the fold, closes the round as `applied`, and opens the next round with the new digests.

`end-round --discard-result` is needed only when the stored result itself is bad. It ends the round with no decision and throws the result away. The seats of that round are not charged, no failure is counted, and they are first in the next round; a batch that would have won must win again. Tell the coordinator before you use it.

### Seats to the front by hand: `front-seats`

Use: a hotkey had a turn outside v2 that nobody will judge, for example a v1 turn after the import snapshot. The operator decides that such a turn is not charged. The command sends the waiting seats of the named hotkeys to the front and clears `demoted`. It charges nothing, counts no failure and changes no other seat; other seats move down by the places that the new front seats take.

- **The order.** Inside the front group the order is the registration block, then the UID, as in every zone of the queue. A seat that goes to the front takes its place by its registration block: it can stand before a seat that was at the front already. The dry run prints the order (`front_seats_after`).
- **`demoted`.** A seat at the front is in the front zone, whatever `demoted` says, and nothing takes `front` away from a seat that waits. So the flag changes no turn. The command clears it so that the public queue does not show a front seat as demoted, and so that the seat does not come back as demoted when a chain sync removes it and adds it again with the same UID and registration block.
- **The squat trim** does not look at a front seat ("The value `squat_min_wait_s`"): it moves only seats of the normal zone.
- **All or nothing.** A hotkey with no seat, a seat in an open round, an open round, or an empty reason: `refused: V2Error: nothing was changed: ...`, exit code 2, and nothing is written. Only the real run writes: a dry run, a refused run and a run with nothing to change do not take over the journal.

**The command writes a new kind of journal record, `seats_fronted`. A checkout without commit `05ed583a` does not know this kind and refuses the whole journal.** So each host that reads the journal goes to the new commit first: AffineValidator (the weights writer and the links job), then the GPU box. The pull of the site and the miner tool read no journal: only public files.

The commit is the one that the coordinator names. It contains `05ed583ad05878b99bc6158a154b4d2444cf17f3`, and it is a checkout only: no install line, no digest block, no `set-params`.

**a. AffineValidator first.** In the shell of "AffineValidator" (the variables of its list):

```
systemctl is-active sn120-v2-weights.service sn120-v2-links.service     # no line says "activating"
cd "$REPO" && git fetch origin && git checkout <the commit>
git merge-base --is-ancestor 05ed583ad05878b99bc6158a154b4d2444cf17f3 HEAD; echo "exit=$?"
"$PY" -c "from sn120_v2.state import HANDLERS; print('seats_fronted' in HANDLERS)"
```

- The third line prints `exit=0`: the checkout contains the commit.
- The fourth line prints `True`: this checkout knows the record kind. **These two lines are the proof before the run on the GPU box.** An older checkout prints `False`.
- The units start a new process at each timer run, and the install of step 1 is editable: the checkout is in force at the next run, with no install and no restart.

Then check that both jobs replay the journal with this checkout:

```
cd "$REPO/v2" && source .venv/bin/activate
( set -a; . "$WENV"; set +a
  python -m sn120_v2.weights_writer --state-dir "$WSTATE" --from-store --network finney --dry-run )
sudo systemctl start sn120-v2-links.service
journalctl -u sn120-v2-links.service -n 5 --no-pager
```

- The dry run is the command of step 5: it writes nothing to the chain. stderr says `kings: read from the journal copy in the store`, and the JSON line has `"status": "would set"`. Do not start the weights unit by hand for this check: a run inside the rate limit uses the attempt of the hour (step 9).
  - **The proof of the replay is this stderr line, word for word: `kings: read from the journal copy in the store`.** The JSON line then has `"kings_from": "the journal copy in the store"`; this key is there only when the state loaded.
  - This command has no `--wallet`, so it does not check the rate limit: it prints `"status": "would set"` also right after a write of v1's writer. The dry run with `--wallet` (step 6) can print `"status": "would defer"` with `recent write: the rate limit allows the next one in N blocks`. The writer loads the state before that check, so the two lines above are printed then too, and the replay is proven. No wait is needed.
  - No proof: `kings: read from the cache of the journal copy, because ...`. The writer then replayed its own older copy, because the store did not answer or holds fewer records. Run the command again until the line names the store.
  - The state did not load: stderr says `kings: not read: <cause>`, and the JSON line is `"status": "would defer"` with the reason `the state failed to load: <cause>`, with no `weights` and no `kings_from`. For a checkout that does not know the record kind, the cause is `unknown journal record kind 'seats_fronted'`.
  - No `kings:` line at all, and a reason that ends with `nothing submitted, the last valid weights stay`: the chain did not answer at the start. This proves nothing: run the command again.
  - **The exit code is 0 in each of these cases.** Read the lines, not the exit code.
- The links job is the command of step 11, the same run as a timer run. The log shows `links: issued N of M waiting hotkeys (...), removed R, valid 86400 s`.
- **What this proves before the run on the GPU box:** the new checkout runs, and both jobs read the journal as it is now. It does not prove that they know the new kind: the journal holds no such record yet. The lines `exit=0` and `True` above prove that.

**e. When the checkout on AffineValidator is safe.** The weights writer starts at minute 05 of each hour; the links job at minute 10 and minute 40; the pull of the site at second 30 of each minute. Do the checkout at a minute from 15 to 35, or from 45 to 00, and a few seconds after second 30. The `is-active` line must not say `activating` (a run is in progress). A pull that starts in the middle of the checkout can fail once: its next run is one minute later.

**b. The GPU box, between two rounds.** The frame is "A checkout only: a running box goes to a later commit": its steps 1 to 3 (no round is open, the stop of the loop, the checkout with the `diff` line), then this command, then its step 4 (the start).

1. Steps 1 and 2 of "A checkout only". After the stop, `status` must show `"phase": "IDLE"`. With an open round the command refuses.
   - `RUNNING` with no box result in the store: `python -m sn120_v2 end-round --state-dir $STATE --stack sn120_v2.assembly:stack` ends the round with no decision. Nobody is charged, and the seats of that round go to the front by this alone. A named seat of that round is then at the front already, and the command only clears its `demoted`.
   - `RUNNING` with a box result in the store, or `APPLY`: start the loop, let `run` close the round, and stop the loop again after the round summary.
2. Step 3 of "A checkout only" (the `diff` line prints only `exit=0`), then:

   ```
   git merge-base --is-ancestor 05ed583ad05878b99bc6158a154b4d2444cf17f3 HEAD; echo "exit=$?"
   ```

3. **The dry run.** One `--hotkey` for each hotkey; the reason goes into the journal.

   ```
   python -m sn120_v2 front-seats --state-dir $STATE --stack sn120_v2.assembly:stack \
     --hotkey <hotkey 1> --hotkey <hotkey 2> \
     --reason "<why these turns are not charged, and who decided it>" --dryrun
   ```

   It reads the journal copy in the store and writes nothing, to the journal or to the store. stdout is one JSON object:
   - `"dry_run": true`, `"written": false`, `"charged": 0`.
   - `seats`: one row for each hotkey, with `uid`, `registration_block`, and `position`, `front` and `demoted`, each with `before` and `after`. `position` is the place in the turn order, from 0, as `position` in `mining.json`.
   - `seats_to_change`: the count of rows that the real run changes. `other_seats_whose_position_shifts`: the count of other seats that move down.
   - `front_seats_after`: the UIDs of the front group after the run, in turn order.

   stderr says `dry run: nothing was written ...` and one reminder: every host that reads the journal must be on this commit before the real run, and after the real run no host can go back to an older commit. Read each row: the right UIDs, `front` `false` to `true`.
4. **The real run:** the same line without `--dryrun`. It takes over the journal, as `set-params` does; that is why the loop is stopped. It prints the same object with `"dry_run": false`, `"written": true`, `"journal_record": N` and `"replay_matches": true`; stderr says `written: journal record N (seats_fronted). ...`. Exit code 0. `"replay_matches": false` and exit code 1: do not start the loop, and tell the coordinator.
5. **The check.** Run the dry run line again: `"seats_to_change": 0`, each row has `front` and `demoted` the same before and after, and stderr says `nothing to change: ...`. `python -m sn120_v2 status --state-dir $STATE` shows `"phase": "IDLE"`, and `"queue"` and `"kings"` as before the run.
6. Start the loop (step 4 of "A checkout only") and do its steps 5 and 6.

**c. After the run.** On AffineValidator, run the two checks of part a again (the dry run of the weights writer, and one run of the links job). They print the same lines; now the journal holds the new record, so this is the proof that both jobs replay it. Then the public queue, about one minute after the first pass of the new `run`:

```
curl -fsS https://affine.io/mining.json | python3 -c "
import json, sys
for row in json.load(sys.stdin)['queue']['next']:
    if row['front']: print(row['position'], row['uid'], row['hotkey'], 'front', row['front'], 'demoted', row['demoted'])"
```

It prints the front group in turn order: the UIDs of `front_seats_after`, each with `front True`. Each seat that the command named shows `demoted False`. A seat that came to the front from a round with no decision keeps its `demoted` value: such a row can show `front True demoted True`, and that is correct (UID 929 on 14 Nov 2026). `demoted` changes no turn of a front seat.

**d. The way back, and its limit.**

- Before the real run: nothing is written. Each host can go back to its commit before.
- After the real run: **no host that reads the journal can go back to a commit without `05ed583a`.** No command removes the record or takes a seat out of the front. A seat leaves the front when it has its turn. A seat that went to the front by mistake: tell the coordinator.
- What an older checkout does after the real run, and the repair (check out the commit again):
  - The GPU box: `run`, `status` and each other command end with `refused: JournalCorrupt: unknown journal record kind 'seats_fronted'`, exit code 2. Nothing is written.
  - The weights writer: each run defers with `the state failed to load: unknown journal record kind 'seats_fronted'`. The last valid weights stay on the chain.
  - The links job: each run ends with `issue failed: unknown journal record kind 'seats_fronted'`. The link files stay; a link is valid for 24 hours, so after that time a miner has no valid link.

### A new GPU box

A new box can have other digests than the old one. `run` on the new box sees that by itself and is refused before it starts anything ("The digests of the box and the pinned digests"). Step 4 shows it before the start, and step 5 says what to do.

1. Stop the loop on the old box, if the box still answers (step 10.4 of "GPU box"). With form B use `sudo systemctl disable --now sn120-v2-run`, so that a reboot of the old box does not start it.
2. On the new box: the checkout, the install, and step 2 of "GPU box" with the same store settings. From the run records (step 11 of "GPU box"), put the `plan.json` of the old box at `$CUT/plan-old-box.json` and the pinned `params.json` at `$CUT/params.json`. Then the first two lines of the digest block (check 3 of "Before the mainnet bootstrap"): they write `$CUT/plan.json` for this box and set `SN120_IMAGE_DIGEST`.
3. The state directory of the new box is new and empty: `mkdir -p $STATE`. Copy nothing into it. Without the directory the loop does not start: it prints `the state directory ... does not exist` and ends with exit code 64.
4. Compare the digests, and look at the state in the store. On a new box, `status` without `--stack` reads the empty local directory.

   ```
   diff $CUT/plan.json $CUT/plan-old-box.json; echo "exit=$?"     # exit=0 and no other output: the same digests
   python -m sn120_v2 status --state-dir $STATE --stack sn120_v2.assembly:stack     # "phase" and "params_hash"
   python -c "import os; from sn120_v2.params import load_params; p = load_params(os.environ['CUT'] + '/params.json'); print(p.squat_min_wait_s, p.params_hash)"
   ```

   The last line must print `31536000` and the `"params_hash"` of `status`: then `$CUT/params.json` is the pinned file. With another hash, the file of the records is not the pinned one: tell the coordinator before step 5. The `plan.json` of the old box holds the pinned digests only if no `set-params` with other digests came after it. If it does not, `run` is refused at step 6 and its line shows the pinned digests; no seat pays for it.
5. **The same digests:** go to step 6. **Other digests:** put the two digests of this box into `$CUT/params.json` first. These are the last three lines of the digest block of "After a fix":

   ```
   cp $CUT/params.json $CUT/params-pinned.json
   python -c "import json,os; c=os.environ['CUT']; plan=json.load(open(c+'/plan.json')); p=json.load(open(c+'/params-pinned.json')); p['recipe']['image_digests']=[plan['image_digest']]; p['recipe']['code_digest']=plan['code_digest']; json.dump(p, open(c+'/params.json','w'))"
   python -c "import os; from sn120_v2.params import load_params; p = load_params(os.environ['CUT'] + '/params.json'); print(p.squat_min_wait_s, p.params_hash)"     # 31536000 and a new hash
   ```

   Never write that file by hand. What comes next depends on the phase that step 4 showed:
   - `IDLE`: `set-params` with `$CUT/params.json` (the `IDLE` line of "After a fix", with its check of `changed`), then step 6.
   - `RUNNING` (the old box died in a round): `end-round` by hand. It works with the old pinned digests on the new box: it pulls the journal copy into the empty state directory. `set-params` before it is refused: `refused: V2Error: a round is open; set the parameters when it has closed (a decided round takes new digests only)`. Exit code 0: the round is ended with no decision; nobody is charged, no failure is counted, and the seats are first in the next round. Then the `IDLE` line. Exit code 3: the dead box stored its result. Then do "A stored result that is not judged, and a box with other digests" of "After a fix" from its step 3: the loop (it judges the result, then `run` is refused), `set-params`, the loop again. It costs nobody anything.
   - `APPLY` (the old box left a decided round): `set-params` with `$CUT/params.json` and the same check of `changed`, then step 6. `run` on the new box then finishes the promotion and the fold.
6. Start the loop (step 10 of "GPU box"). With the same digests, its `end-round` pulls the journal copy from the store and ends the open round of the dead box with no decision; a stored box result is kept, and `run` judges it. Then `run` continues. Do not run `bootstrap` again.

Two `run` processes never both write: the newer one takes over the journal, and the older one stops with `refused: FencedError` at its next write. The loop of the old box does not start a refused `run` again: it stops with exit code 2. Never start it again by hand on the old box.

### Go back

Before the first v2 round ends. The weights part is "To go back" of the handover: stop v2's timer first, then start v1's writer.

1. **Stop v2's writers.** Stop the loop on the GPU box (step 10.4 of "GPU box"; with form B, `sudo systemctl disable --now sn120-v2-run`). On AffineValidator:

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

**Why `affine-ff.service` must be stopped before the first run.** The v1 service and this job write the same files in the same directory. Every 30 minutes v1 writes `ff/links/<hotkey>.json` and `ff/links.json`, and removes the file of each hotkey that is not eligible in v1's view. v1's view is not v2's queue: v1 gives a link to each activated hotkey that waits for a v1 turn, and none to a hotkey that had its turn. There is no lock between the two. If both run:

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

**Group A: the round path. Check that each is inactive and disabled.** The sign that anybody can read: a new row in `turns` of `https://affine.io/k1.json` about every 15 minutes means that the lifecycle runs. On 14 Nov 2026 it did ("The state of v1" has the line that prints the rows). The sign works only while `affine-k1.service` runs. After its stop the proof is the unit state: `systemctl --user is-active affine-mimo-lifecycle.service` prints `inactive`.

1. The watchdogs: `tp-watchdog`, `ff-watchdog`, and any other unit that `systemctl --user list-units '*watchdog*'` shows. A watchdog reverts v1 or starts it again when the lifecycle is down for 90 seconds (`ops/free_file/README.md`, `ops/ff_train_params/README.md`). The points below are from v1's code in this repository, not from the host.
   - **The result to expect: the list shows no watchdog that reverts v1.** Each watchdog is started for a fixed time or for a fixed count of epochs. Then it ends by itself with exit code 0, and nothing starts it again (its unit has `Restart=on-failure` or no `Restart=`).
     - `ff-watchdog`: until 31 Oct 2026, 9:00 AM Costa Rica time (`WATCH_UNTIL=1793458800` in `activate_ff.sh` and `activate_tp.sh`). Its last log line is `watch window over; exiting`.
     - `tp-watchdog`: 24 hours. `12k-watchdog`: 3 days. `sm-watchdog`: 7 days.
     - The watchdogs of the gate (`affine-vl-watchdog`, `affine-cap-watchdog`, `affine-zflat-watchdog`, `affine-prior-watchdog`, `affine-irt-watchdog`): 24 to 72 hours.
     - `ts-watchdog`, `ts-watchdog-f` and `legacy-watchdog`: 2 or 3 trainings or epochs.
     - The newest commit that starts one of them is of 2 Nov 2026, so the longest window ended about 9 Nov 2026.
   - `affine-perf-watchdog.timer` and its service can be in the list. They revert nothing: group C has them.
   - **If another watchdog is listed** (somebody armed it again by hand, or a change that is not in the repository): `systemctl --user stop <unit>` first, before any unit of group A, then `systemctl --user disable <unit>`. `systemd-run` started these units, so `disable` can print an error for a unit that has no unit file; the proof is that the list does not show the unit any more.
   - **Why first.** `ff-watchdog` runs the revert of `ops/free_file/activation/activate_ff.sh` when the lifecycle is not active for 90 seconds, when no epoch is finalized for 45 minutes, or when the selection failed in 2 epochs. The revert starts `affine-mimo-lifecycle.service` again. Then it runs `systemctl --user restart` on `affine-training-params`, `affine-k1`, `affine-k1-transparency`, `affine-mining-discovery` and `affine-promotion-gate`. A restart also starts a unit that is stopped and disabled. So a stopped unit of group A or group C comes back, and v1's `mining.json` goes over v2's.
2. `affine-mimo-lifecycle.service`: selects the next turn and starts the training. From v1's code in this repository:
   - It gives turns without `affine-ff.service`: it takes the hotkeys from the chain file of `affine-k1.service` (`K1-CHAIN.json`, written every 300 seconds) and lists the miners' files in the bucket by itself. Only its own stop ends the epochs.
   - When `affine-k1.service` is stopped and the lifecycle still runs, the chain file gets too old (the limit is a value on the host). Then each selection ends as a skip with the reason `ff-selection-error`: v1 gives no more turns, but the lifecycle still makes epochs, and `k1.json` no longer shows them.
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
- `affine-prior-table.service`, `affine-perf-eval.service`, `affine-perf-eval-2.service`, `affine-perf-watchdog.timer`, `affine-bench.service`. The timer runs `ops/performance_eval/perf_watchdog.sh` every 5 minutes: it restarts `affine-perf-eval` and `affine-perf-eval-2` when they are not active and still enabled. Disable these two, or stop the timer first; `stop` alone does not hold.
- `affine-site-baseline.timer`, `affine-site-k1-tables.timer`, `affine-site-slices.timer`, `affine-site-metrics.service`.

Leave `sn120-disk-guard.timer` on. `sn120-pod-watch.timer` only reports the state of the v1 pods and never rents (`ops/pods/README.md`); stop it when the v1 pods are released. Do not run `replace_pod.py replace` for a v1 role after the cutover.

## Store

v2 and v1 share one bucket, and v2 has no prefix. The rule: **no v2 write and no v2 delete reaches a v1 object.** The preflight ("GPU box", step 5) checks it on the bucket before the first write.

No top-level prefix belongs to v2 alone, except `pending/`. v1 has objects under `public/` and under `private/`. So v2's keys are exactly the ones below, and nothing lists, copies or deletes a whole `public/` or `private/`.

**What v2 writes** (`sn120_v2/layout.py`). With the miner files below, these are the 16 lines of the preflight:

| Key or prefix | Writer |
| --- | --- |
| `public/head.json` (the pointer to the current head), `public/heads/` | controller |
| `public/llms.txt`, `public/mining.json` | publisher |
| `public/rounds/`, `public/loras/`, `public/irt/` | controller |
| `public/bases/`, `public/masters/` | the genesis upload, then the controller after each fold |
| `private/journal/`, `private/rounds/` | controller, admission |
| `pending/` | the box, for every result before the controller has checked it |
| `public/benchmarks.json` | the benchmark service, when it exists |
| `public/ff/links/`, `public/ff/links.json` | the links job, only when it runs with `--to-store`. At this cutover it runs with `--site-dir` and writes nothing to the bucket. The preflight still checks these two. |

**What v2 reads and never writes:** `private/ff/v1/miners/<hotkey>/batch.json`, the miner's own file. The key is the same as in v1, on purpose. Only the miner writes it, through its upload link.

**What v2 deletes.** Only objects that v2 wrote itself:

- everything under a round's `pending/` prefix (`pending/rounds/<round>`), when the round closes;
- everything under the staging prefix of a fold (`pending/folds/<head>`), after the fold. It is under `pending/` too, and only v2's fold writes there. The delete refuses every prefix that is not one round or one fold under `pending/`;
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
| `--k1`, `--mining` | Another URL or a local file. The defaults are the two public URLs. Needed once the site serves v2's `mining.json` ("GPU box", step 7). |

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
