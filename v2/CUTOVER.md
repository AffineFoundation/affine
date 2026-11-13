# Cutover: start v2 on mainnet from v1's state

This is the operator's runbook for the mainnet cutover. Do the smoke test first: [SMOKE.md](SMOKE.md). Cut over only after it passes.

The operator does every step here. No step is automatic, and no worker touches a host, a wallet or the chain.

## Fixed inputs (Jacob, 13 Nov 2026)

| Input | Value |
| --- | --- |
| Validator hotkey (sets weights from v2, as in v1) | `5HmYnmUYT6qe3yFMg1Ad8WLLqDvnwtjYakXBpDvoRW1Qqzb8` |
| Its coldkey | `5Fc3ZZQAYB3SPXKcFnd1WJeyQvArSZZeB6LU1rb7zvQ6XvDh` |
| Subnet owner hotkey (`--owner-hotkey`) | The hotkey of UID 0. v1's `ops/HOURLY_BURN.md` names the same address as the validator hotkey. The chain is the truth: storage item `SubnetOwnerHotkey` of subnet 120. |
| v1 turns with no verdict | To the front of the v2 queue, not charged (`--undecided front`, the default). |
| Genesis model | `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`, fixed in the code. |

Both addresses are public. The wallet stays on the validator host. No command here reads it, except the weights writer in step 11.

## Code to check out

Check out the commit that the coordinator names for the cutover: the tip of `main` after `v2` took it over. The v1 code is on the archive branch and the tag that the coordinator made before that. Do not run the cutover from a worker branch or from an older commit.

```
git fetch origin && git checkout <commit that the coordinator names>
git rev-parse HEAD        # write this down with the cutover records
cd v2 && uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,gpu,r2,chain,miner]" && python -m pytest
```

## What the import carries

`python -m sn120_v2.cutover` reads two public files, `https://affine.io/k1.json` and `https://affine.io/mining.json`. It needs no credential, reads no v1 host and no bucket, and writes nothing to the chain. It writes:

- `genesis.json` (`GenesisSpec`): the fresh MiMo genesis head and the validator hotkey.
- `v1_import.json` (`V1Import`): v1's last 10 kings; each registration and hotkey that had its turn in v1; the seats v1 moved to the back; the seats that are owed a turn at the front.
- `chain.json`: a copy of the chain snapshot that you gave.
- A copy of each v1 file that it read (the evidence).

At the start, v2 builds its queue from the chain: each registration that is not in the served lists waits, in order of registration block, then UID. Thus each paid registration gets one turn, and no registration gets two. The kings are v2's first kings, oldest first, so the first v2 weights pay the same 10 miners that v1 pays.

A queued miner file is not in the import. v2 reads it at the same key as v1 (see "Store"). A v1 file with train keys that only v1 had is invalid in v2: the hotkey keeps its place and is not charged until the miner uploads a new file.

## Steps

### Before the cutover

1. **Save copies of `k1.json`.** `k1.json` shows only v1's last 200 epochs. A turn that v1 took and did not judge is lost when it leaves that window: the miner is then charged with no verdict. Save a copy now, then one each 12 hours. Keep all copies.

   ```
   curl -fsS https://affine.io/k1.json -o k1-$(date -u +%Y%m%dT%H%M%SZ).json
   ```

   The first copy is in this repo: `tests/data/cutover/k1.json` (13 Nov 2026, 14 turns with no verdict: epoch 1157 and epochs 1288 to 1310).
2. **Build the genesis files** (57 GB of disk, 30 GB of RAM, no GPU). The last output line is JSON with `master_digest`. The command also writes `<genesis dir>/genesis_files.json`. With `--upload` it puts the base and master files in the store that `SN120_R2_*` names, so set those variables to the mainnet store first.

   ```
   python -m sn120_v2.gpu.genesis --out <genesis dir> --upload
   ```

### Stop v1's rounds

3. **Stop v1's round path** on the v1 host (`systemctl --user`). v1 must not start a turn after the import. Use the order in "v1 units", group A. Keep v1's hourly weight writer on until step 11.
4. **Save the last `k1.json`** with the command of step 1, after v1's round path is stopped. `affine-k1.service` must still run: the import refuses a `k1.json` that is older than one hour.
5. **Write a chain snapshot**, after the last v1 files. It needs the `chain` extra and no wallet.

   ```
   python -m sn120_v2.chain --netuid 120 --network finney --out chain_snapshot.json
   ```

### Import

6. **Dry run.** It prints the import and writes nothing. Give `--earlier-k1` one time for each saved copy. Run it two times; the kings and the seats must be the same.

   ```
   python -m sn120_v2.cutover --dry-run --chain chain_snapshot.json \
       --owner-hotkey 5HmYnmUYT6qe3yFMg1Ad8WLLqDvnwtjYakXBpDvoRW1Qqzb8 \
       --earlier-k1 tests/data/cutover/k1.json --earlier-k1 <later copy>
   ```

7. **Read the output.**
   - The 10 kings are the miners that v1 pays now.
   - The front seats are the turns that v1 did not judge. Each line shows how many turns that hotkey had in v1. On 13 Nov 2026, four of the 14 hotkeys had 13 to 15 turns each.
   - `first weights` must show 10 UIDs.
   - The last line must say `UID 0 is <address> ... it is the given owner hotkey`.
8. **Real run.** Add the genesis inputs, and replace `--dry-run` with `--out <new directory>`.

   ```
   python -m sn120_v2.cutover --out <cutover dir> --chain chain_snapshot.json \
       --master-digest <master_digest of step 2> \
       --validator-hotkey 5HmYnmUYT6qe3yFMg1Ad8WLLqDvnwtjYakXBpDvoRW1Qqzb8 \
       --owner-hotkey 5HmYnmUYT6qe3yFMg1Ad8WLLqDvnwtjYakXBpDvoRW1Qqzb8 \
       --earlier-k1 tests/data/cutover/k1.json --earlier-k1 <later copy>
   ```

   The command never replaces a directory. For a second run, use a new directory.
9. **Miner files.** Do this only if v2's store is not v1's bucket with no prefix. See "Store" for the command and for the choice.

### Start v2

10. **First start.** `--stack` names the builder of the real parts, as in SMOKE.md. The command refuses a second run on the same state directory.

    ```
    python -m sn120_v2 bootstrap --state-dir <state> --params <params.json> --stack sn120_v2.assembly:stack \
        --genesis <cutover dir>/genesis.json --genesis-files <genesis dir>/genesis_files.json \
        --v1-import <cutover dir>/v1_import.json --chain <cutover dir>/chain.json
    ```

    It prints the summary. `kings` must be the 10 UIDs of step 7, and `queue` must be the number of seats that step 7 printed.
11. **Move the weights writer from v1 to v2, in one step.** Only one service can set weights.
    1. Dry run of the v2 writer. It needs no wallet and writes nothing. The table must have the 10 UIDs of step 7, and `validator_registered` must be `true`.

       ```
       python -m sn120_v2.weights_writer --state-dir <state> --network finney --params <params.json> --dry-run
       ```

    2. Stop and disable v1's hourly weight writer ("v1 units", group B). Make sure that no v1 writer is left.
    3. Start v2's hourly timer with this command. `NAME/HOTKEY` is the wallet of the validator hotkey above.

       ```
       python -m sn120_v2.weights_writer --state-dir <state> --network finney --params <params.json> --wallet NAME/HOTKEY
       ```

    Never enable `affine-hourly-burn.timer` or `affine-transition-weights.timer`. If the v2 writer cannot make a table, it sets nothing and the last weights stay on chain.
12. **Start the round loop.**

    ```
    python -m sn120_v2 run --state-dir <state> --params <params.json> --stack sn120_v2.assembly:stack
    ```

13. **After the start.** Stop the remaining v1 units ("v1 units", group C) when the v2 publisher and the v2 website replace them. Put `<cutover dir>` in the durable bucket with the v2 run records.

To go back before the first v2 round ends: stop the v2 writer and the round loop, then start v1's writer again. v1's state did not change, because the import only read public files.

## v1 units

The names come from documents in this repository. The repository does not show which of them run now. The host is the truth: `systemctl --user list-units 'affine-*' 'sn120-*' '*watchdog*'` and `systemctl --user list-timers --all`.

**Group A: the round path (step 3), in this order.**

1. The watchdogs: `tp-watchdog`, `ff-watchdog`, and any other unit that `systemctl --user list-units '*watchdog*'` shows. Stop them first. A watchdog reverts v1 or starts it again when the lifecycle is down for 90 seconds (`ops/free_file/README.md`, `ops/ff_train_params/README.md`).
2. `affine-mimo-lifecycle.service`: selects the next turn and starts the training. Stop it when the trainer is idle, as v1's own activation scripts do (`ops/free_file/activation/activate_ff.sh`).
3. `affine-promotion-gate.service`: judges a turn. After this stop, the set of turns with no verdict does not change.
4. `affine-mimo-trainer-supervised.service` and `affine-mimo-evaluator-supervised.service`, with their `-tunnel` units (`ops/pods/roles.json`).
5. `affine-ff.service`: issues the miners' upload links and checks their files. Stop it when v2 issues upload links. A v1 link stays valid for up to 24 hours.

**Group B: the weights writer (step 11).**

- The repository names the hourly writer in two ways. `docs/MIMO_DEPLOYMENT_RUNBOOK.md` names `affine-live-reward-writer.timer` and `affine-live-reward-writer.service`. The K1 service (`ops/k1_pass_rates/services/k1_service.py`) says that the hourly writer is `weights_flip.py`, which reads `website/scores.json`; no document gives the unit name of that script.
- So find the writer on the host: `systemctl --user list-timers --all`, and `grep -rlE 'weights_flip|live_reward_writer|current_assessment_writer|set_weights' ~/.config/systemd/user/`. Stop and disable each timer and service that these show.
- `affine-hourly-burn.timer` and `affine-transition-weights.timer` are old writers (`ops/HOURLY_BURN.md`, `ops/EQUAL_REGISTRATION_WEIGHTS.md`). They must stay disabled and masked.
- `affine-k1.service` writes `website/scores.json` and `website/k1.json`. It does not write to the chain. Stop it after v1's writer is stopped and the last `k1.json` is saved.

**Group C: publishers and side services (step 13).** Stop each one when its v2 replacement runs; never run two writers of one public file.

- `affine-mining-discovery.service` (`mining.json`), `affine-k1-transparency.service`, `affine-training-params.service`.
- `affine-prior-table.service`, `affine-perf-eval.service`, `affine-perf-eval-2.service`, `affine-perf-watchdog.timer`, `affine-bench.service`.
- `affine-site-baseline.timer`, `affine-site-k1-tables.timer`, `affine-site-slices.timer`, `affine-site-metrics.service`.

Leave `sn120-disk-guard.timer` on. `sn120-pod-watch.timer` only reports the state of the v1 pods and never rents (`ops/pods/README.md`); stop it when the v1 pods are released. Do not run `replace_pod.py replace` for a v1 role after the cutover.

## Store

v2 writes these keys (`sn120_v2/layout.py`): `public/current.json`, `public/llms.txt`, `public/mining.json`, `public/benchmarks.json`, `public/heads/`, `public/rounds/`, `public/loras/`, `public/bases/`, `public/masters/`, `public/irt/`, `private/rounds/`, `private/journal/`, `pending/`, and it reads `private/ff/v1/miners/<hotkey>/batch.json`. It deletes only under `pending/`.

What v1's source shows for the same bucket with no prefix:

- `public/current.json` is the only v1 key that v2 overwrites. v1 calls it "an old mock under a different authority" (`docs/MIMO_DEPLOYMENT_RUNBOOK.md`); v1's live pointer is `public/streams/nonpayable-live-reward-math-v1-/current.json`.
- `private/ff/v1/miners/<hotkey>/batch.json` is the same key on purpose. Miners overwrite it in v1 too. v1's record of a turn is a separate copy under `private/ff/v1/turns/<epoch>/`, which v2 never writes.
- v1 serves `llms.txt`, `mining.json`, `k1.json`, the turn batches and the fp32 master records from the website directory, not from the bucket. v2 writes none of v1's history prefixes (`public/checkpoints/`, `public/streams/`, `public/epoch-evidence/`, `private/ff/v1/turns/`).

This comes from the source and the public files only. Nobody listed the bucket for this check.

**Recommendation: a prefix, plus a copy of `private/ff/v1/miners/`.** Set `SN120_R2_PREFIX` (for example `v2`). Then no v2 process can write or delete a v1 object, and v1's bucket stays as the archive for the website. The cost is one copy, and one rule for miners: a file that a miner uploads through an old v1 link after the copy is not seen by v2. Say that in the announcement, and stop `affine-ff.service` before the copy.

The copy reads v1's location and writes only under v2's prefix. Use a read-only token for v1's bucket. `limit=None` copies the file of each waiting hotkey, in queue order.

```
export CUTOVER_DIR=<cutover dir>
export V1_R2_ENDPOINT=... V1_R2_BUCKET=... V1_R2_ACCESS_KEY_ID=... V1_R2_SECRET_ACCESS_KEY=...   # read-only token
python - <<'EOF'
import os
from pathlib import Path
from sn120_v2.cutover import copy_miner_files, preview_queue
from sn120_v2.params import Params
from sn120_v2.r2 import R2Store
from sn120_v2.types import ChainSnapshot, V1Import

env, out = os.environ, Path(os.environ["CUTOVER_DIR"])
assert env.get("SN120_R2_PREFIX"), "set SN120_R2_PREFIX: this copy never writes to v1's keys"
imp = V1Import.from_json((out / "v1_import.json").read_bytes())
snapshot = ChainSnapshot.from_json((out / "chain.json").read_bytes())
hotkeys = [seat.reg.hotkey for seat in preview_queue(imp, snapshot, Params())]
src = R2Store(endpoint=env["V1_R2_ENDPOINT"], bucket=env["V1_R2_BUCKET"],
              access_key_id=env["V1_R2_ACCESS_KEY_ID"], secret_access_key=env["V1_R2_SECRET_ACCESS_KEY"])
print("\n".join(copy_miner_files(src, R2Store.from_env(), hotkeys, limit=None)))
EOF
```

Each line names one copied file, its sha256, and `default recipe` when the file has no `train` object. For the smoke test, use the smoke prefix and `limit=8`.

## Options of the import command

| Option | Meaning |
| --- | --- |
| `--dry-run` or `--out <dir>` | Exactly one. Print only, or write the records to a new directory. |
| `--master-digest` | sha256 digest of the genesis fp32 master files (step 2). No default. |
| `--validator-hotkey` | Public SS58 address of the validator hotkey. A wrong character is refused. No default. |
| `--undecided front` or `served` | For turns that v1 took and did not judge. `front` (default, Jacob's rule): not charged, first in the v2 queue. `served`: charged, no new turn. |
| `--earlier-k1 <file>` | A saved earlier copy of `k1.json`. Give it one time for each copy. A turn with no verdict that only an earlier copy shows gets the `--undecided` rule. If a newer copy shows the verdict of a turn, the turn stays judged. |
| `--chain <file>` | A `ChainSnapshot` as JSON (step 5). Adds checks, and prints the start queue, the first weights and the hotkey of UID 0. `--out` copies it as `chain.json`. |
| `--owner-hotkey` | The subnet owner hotkey. The command refuses when UID 0 of `--chain` has another hotkey. Needs `--chain`. |
| `--max-age-s` | Oldest `k1.json` accepted, in seconds. Default 3600. Earlier copies have no age limit. Use a larger value only if v1's publisher stopped, and then the smallest value that accepts the file. |
| `--k1`, `--mining` | Another URL or a local file. The defaults are the two public URLs. |

## When the command refuses

It exits with code 2, prints `CUTOVER IMPORT REFUSED` and the reason, and writes nothing. It refuses when:

- a v1 file cannot be read, has an unknown version or lacks a field;
- `k1.json` is older than `--max-age-s`, or `mining.json` is from another epoch or shows a head that `k1.json` does not know;
- the served lists, their counts and the turn history do not agree, or a waiting seat is also listed as served;
- the payees are not the last 10 distinct GREEN turns (a `pending` or timeout verdict is never a king);
- an earlier copy is newer than `k1.json`, or shows a turn that is not in v1's served turns;
- a retry grant names a hotkey that had no turn;
- with `--chain`: the snapshot is older than v1's chain read, UID 0 is absent, the validator hotkey is not registered, or no king is registered (v2 would send all weight to UID 0);
- with `--owner-hotkey`: UID 0 has another hotkey, or `--chain` is absent.

Do not edit a v1 file to make the command pass. Find the cause in v1, or ask.
