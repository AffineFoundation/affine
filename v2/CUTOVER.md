# Cutover: start v2 from v1's state

Do the smoke test first: [SMOKE.md](SMOKE.md). Cut over only after it passes.

One command builds the two records that v2 starts from. It reads two public files, `https://affine.io/k1.json` and `https://affine.io/mining.json`. It needs no credential, reads no v1 host and no bucket, and writes nothing to the chain.

- `genesis.json` (`GenesisSpec`): the fresh MiMo genesis head and the validator hotkey.
- `v1_import.json` (`V1Import`): v1's last 10 kings; each registration and hotkey that had its turn in v1; the seats v1 moved to the back; the seats that are owed a turn at the front.

v2 builds its queue from the chain: each registration that is not in the served lists waits, in order of registration block, then UID. Thus each paid registration gets one turn, and no registration gets two.

## Steps

1. **Save copies of `k1.json` until the cutover.** `k1.json` shows only v1's last 200 epochs. A turn that v1 took and did not judge (a pending turn) is lost when it leaves that window: the miner is then charged with no verdict. Save a copy now, then one each 12 hours, and one before step 4. Keep all copies.

   ```
   curl -fsS https://affine.io/k1.json -o k1-$(date -u +%Y%m%dT%H%M%SZ).json
   ```

   The first copy is in this repo: `tests/data/cutover/k1.json` (13 Nov 2026, 14 pending turns, the oldest at epoch 1157).
2. **Stop v1's round service** (Jacob, on the v1 host). v1 must not start a turn after the import. Keep v1's hourly weight writer on until step 8.
3. **Write a chain snapshot.** The weights writer ([pull request #102](https://github.com/AffineFoundation/affine/pull/102)) has a read-only command that writes a `ChainSnapshot` as JSON. Use that file as `--chain`.
4. **Dry run.** It prints the import and writes nothing. Give `--earlier-k1` one time for each saved copy. Run it two times; the kings and the seats must be the same.

   ```
   cd v2 && source .venv/bin/activate
   python -m sn120_v2.cutover --dry-run --undecided front --chain <snapshot.json> \
       --earlier-k1 tests/data/cutover/k1.json --earlier-k1 <later copy>
   ```

5. **Read the output.**
   - The 10 kings are the miners that v1 pays now.
   - The front seats are the turns that v1 did not judge. Each line shows how many turns that hotkey had in v1.
   - The line `UID 0 is <address>` shows the subnet owner hotkey. See step 8.
6. **Real run.** Add the genesis inputs, and replace `--dry-run` with `--out <new directory>`. The command writes the two records and a copy of each v1 file that it read (the evidence).

   ```
   python -m sn120_v2.cutover --master-digest <sha256> --validator-hotkey <address> --undecided front \
       --chain <snapshot.json> --earlier-k1 tests/data/cutover/k1.json --earlier-k1 <later copy> --out <new directory>
   ```

7. **Keep the directory.** Put it in the durable bucket with the v2 run records.
8. **Check UID 0, then start v2.**
   - v2 sends weight that no king earned to UID 0, and the weights code does not check who has UID 0. Before the first v2 weights write, write a new chain snapshot and make sure that the hotkey of UID 0 is the subnet owner hotkey. The chain gives the owner hotkey of subnet 120 in the storage item `SubnetOwnerHotkey`. If the two addresses are different, do not start the weights writer.
   - Start v2 from `genesis.json` and `v1_import.json` (see SMOKE.md for the start command), and stop v1's hourly weight writer in the same change. Only one service can set weights.

## Options

| Option | Meaning |
| --- | --- |
| `--dry-run` or `--out <dir>` | Exactly one. Print only, or write the records to a new directory. |
| `--master-digest` | sha256 digest of the genesis fp32 master files. The node computes it when it builds the genesis (SMOKE.md). No default. |
| `--validator-hotkey` | Public SS58 address of the validator hotkey that v1 uses. v2 keeps it. A wrong character is refused. No default. |
| `--undecided front` or `served` | For turns that v1 took and did not judge. `front`: not charged, first in the v2 queue. `served`: charged, no new turn. No default. |
| `--earlier-k1 <file>` | A saved earlier copy of `k1.json`. Give it one time for each copy. A pending turn that only an earlier copy shows gets the `--undecided` rule. If a newer copy shows the verdict of a turn, the turn stays judged. |
| `--chain <file>` | A `ChainSnapshot` as JSON (step 3). Adds checks, and prints the start queue, the first weights and the hotkey of UID 0. |
| `--max-age-s` | Oldest `k1.json` accepted, in seconds. Default 3600. Earlier copies have no age limit. |
| `--k1`, `--mining` | Another URL or a local file. The defaults are the two public URLs. |

## What Jacob must give or decide

- The validator hotkey address, and the wallet for that hotkey on the v2 host (step 8 only; this command never reads a wallet).
- The fp32 master digest, from the genesis build on the node.
- `--undecided`. On 13 Nov 2026, v1 had 14 turns without a verdict. We recommend `front`: those miners paid, and v1 did not judge them. Four of the 14 hotkeys had 13 to 15 turns each in v1 (v1 gave them retries); the dry run shows the count.
- `--max-age-s`, only if v1's publisher stopped and `k1.json` is old. Use the smallest value that accepts the file.
- Miner files. v2 reads them at the same key as v1: `private/ff/v1/miners/<hotkey>/batch.json`. If v2 uses another bucket, copy that prefix with the bucket credentials. This command does not copy objects.
- Stop of v1's round service (step 2) and of v1's weight writer (step 8).

## When the command refuses

It exits with code 2, prints `CUTOVER IMPORT REFUSED` and the reason, and writes nothing. It refuses when:

- a v1 file cannot be read, has an unknown version or lacks a field;
- `k1.json` is older than `--max-age-s`, or `mining.json` is from another epoch or shows a head that `k1.json` does not know;
- the served lists, their counts and the turn history do not agree, or a waiting seat is also listed as served;
- the payees are not the last 10 distinct GREEN turns (a `pending` or timeout verdict is never a king);
- a turn has no verdict and `--undecided` is absent;
- an earlier copy is newer than `k1.json`, or shows a turn that is not in v1's served turns;
- a retry grant names a hotkey that had no turn;
- with `--chain`: the snapshot is older than v1's chain read, UID 0 is absent, the validator hotkey is not registered, or no king is registered (v2 would burn all weight).

Do not edit a v1 file to make the command pass. Find the cause in v1, or ask.
