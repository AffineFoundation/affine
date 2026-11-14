# The check of llms.txt against the params

`llms.txt` is written by the publisher from the same snapshot as `mining.json`. `mining.json` holds the params. This file says how to prove that a published `llms.txt` is true to them, and what to do when it is not.

There are two checks.

1. **In the test suite** (`tests/test_publisher_params.py`). It runs with every change of the code. Nothing to do on a host.
2. **A command**, for the published files. It reads; it changes nothing.

## The command

```
python -m sn120_v2.publisher check --url <public address>
```

It reads `<public address>/llms.txt` and `<public address>/mining.json`, as any browser does. The public address is the value of `SN120_PUBLIC_URL` ([CUTOVER.md](CUTOVER.md), "Hosts").

| Option | Use |
| --- | --- |
| `--url URL` | The public address. |
| `--mining-json FILE --llms-txt FILE` | Two files or two addresses, in place of `--url`. |
| `--state-dir DIR` | Also compare with the params of the validator's state (the journal in `DIR`). Read-only; safe while `run` writes. |
| `--values-only` | Do not compare the whole text. Use it when the code of this host is not the code that published the files. |
| `--tries N`, `--wait-s S` | Read again when the two files are not from the same snapshot (an update is in progress). Default: 3 tries, 5 seconds. |

### What it compares

1. `mining.json` is whole: its content has the `snapshot_digest` that it names, and its params have the `params_hash` and the `recipe_id` that it names.
2. `llms.txt` is the text of that `mining.json`: the same snapshot digest, the same `params_hash`, the same `recipe_id`, and each row of the table "Every parameter" has the value of `mining.json`.
3. With `--state-dir`: the published params are the params of the state.
4. Without `--values-only`: `llms.txt` is, byte for byte, the text that the code of this host renders from that `mining.json`. This covers every sentence.

### Where to run it

| Host | Command | Covers |
| --- | --- | --- |
| Any host with a copy of the branch `v2` (Python 3.11 or later; no package to install) | `python -m sn120_v2.publisher check --url $SN120_PUBLIC_URL --values-only` | Points 1 and 2. |
| The same, at the commit that the GPU box runs | `python -m sn120_v2.publisher check --url $SN120_PUBLIC_URL` | Points 1, 2 and 4. |
| The GPU box, in the directory `v2` of its checkout | `python -m sn120_v2.publisher check --url $SN120_PUBLIC_URL --state-dir $STATE` | All four points. |

The command does not need the GPU box. It needs no restart, no credential and no change of a running service. On the GPU box it is a new process next to `run`; it reads the journal file and opens no store.

### What is right

Exit code 0 and one line:

```
ok: <address>/llms.txt is true to the params of <address>/mining.json (every line; params_hash <64 hex>; generated <time> UTC)
```

With `--values-only` the line says `every stated parameter` in place of `every line`.

### What is wrong, and what to do

Exit code 1, and a list on the error output. Exit code 2: the check did not run (the address gave no answer, or a file is missing).

| Line | Cause | Do |
| --- | --- | --- |
| `llms.txt and mining.json are not a pair` | The two files are from two snapshots. The public pull on AffineValidator copies them one after the other, so this can be true for a moment. | Run again after a minute. If it stays, look at the public pull (CUTOVER.md). |
| `` `<name>`: llms.txt states A, mining.json has B `` | The text states an old value of a parameter. | On the GPU box: `python -m sn120_v2 publish --state-dir $STATE --stack sn120_v2.assembly:stack`. It writes both files again from the state. Then wait for the public pull and run the check again. |
| `the published params are not the params of the state: ... differ` | A `set-params` was done and the files were not published after it. | The same `publish` command. |
| `mining.json: the content does not have the snapshot_digest ...` or `` `params_hash` is not the id of its own params `` | `mining.json` was changed after the publisher wrote it. | The same `publish` command. Then find what changed the file. |
| `llms.txt is not the text that this code renders from mining.json: N lines differ` | The code of this host is not the code that published the files. Or a sentence of the file was changed. | Compare the commit of this host with the commit of the GPU box (`git rev-parse HEAD`). If they differ: use `--values-only`, or check out the commit of the box. If they are the same: the `publish` command. |

`publish` is the command of [SMOKE.md](SMOKE.md), step 7. It reads the journal and does not take it over. It reads the file of every waiting seat again, so it can take some minutes.

### A timer

The exit code makes the command fit for a timer on any host with a copy of the code. An example line for `cron`, each hour. It is an example: it was not run on a host of the subnet.

```
7 * * * * cd <checkout>/v2 && python3 -m sn120_v2.publisher check --url <public address> --values-only || <your alert command>
```

### The way back

The command changes nothing, so there is nothing to undo. To stop a timer, remove its line.

## What the two checks catch

The test suite fails when:

- a parameter or a recipe value changes and the text stays the same (each field is changed alone);
- a new parameter is added and the text does not state it;
- a sentence holds a number that no parameter gives and that is not on the list of the test. Each number on that list stands in a phrase that the test builds from the constant of the code, or it is a formula, a step number, or one of four facts of the chain and of file formats;
- the text from a snapshot and the text from the `mining.json` that was read back differ.

The command fails when a published pair has one of the faults of the table above.

Neither check catches:

- a number that is written as a word ("three", "twice");
- a rule in words that is no longer the rule of the code. The check proves the numbers and the identity of the two files, not the meaning of a sentence;
- a number that the publisher computes from a parameter with a wrong formula. Such a number moves with the parameter, so the test accepts it;
- a difference between the published params and the params that the box really runs, when the command runs without `--state-dir`.
