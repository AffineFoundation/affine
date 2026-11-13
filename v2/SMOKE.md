# SN120 v2 smoke test

This runbook has two parts. Part 1 runs today on any machine. Part 2 is the one real round on one secure 8x B300. Jacob runs Part 2. Nothing here sets chain weights, and nothing here is the cutover.

Words used here:

- **State directory**: the folder that holds the journal (`journal.jsonl`). The journal is the controller's only memory.
- **Stack**: the set of real parts the controller uses (store, admission, sampler, trainer, evaluator, folder, publisher). `--stack module:function` names the function that builds them.
- **Round**: take up to `batch_cap` queued batches, train one LoRA per batch, score every LoRA and the head M, apply at most one winner.

## Part 1: CPU check (about 5 seconds, no GPU, no network)

```
cd v2
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
python -m pytest
python -m sn120_v2 sim --state-dir /tmp/sn120-sim --rounds 6 --miners 4
python -m sn120_v2 status --state-dir /tmp/sn120-sim
```

Pass when:

1. `pytest` reports no failure.
2. `sim` prints `"replay_matches": true` and `"phase": "IDLE"`, and exits with code 0.
3. `status` shows `"head_folded": true` and a list of `kings`.

`sim` uses fake training and fake scoring. It proves the round loop, the journal, crash recovery and the store rules. It proves nothing about GPUs.

## Part 2: one real round on one secure 8x B300

Do this only after the sibling pull requests are merged into `v2`. Two commands below come from other workstreams and are marked **(sibling)**.

Rules for this test:

- Use a new, empty state directory and a store location that production does not use.
- Do not start the weights writer.
- Use the secure tier, not spot.

Steps:

1. **Install on the box.**
   `cd v2 && uv venv .venv --python 3.12 && source .venv/bin/activate && uv pip install -e ".[dev,gpu,r2,chain,miner]"`
2. **Build the genesis files (sibling: train/eval workers).** The command downloads `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`, checks it against base hash `87de8304...`, builds the fp32 master, uploads both, and writes `genesis_files.json` (`{"base_files": {...}, "master_files": {...}}`) and the fp32 master digest.
3. **Make the cutover files (sibling: cutover import).** `python -m sn120_v2.cutover ... --out <dir>` writes `genesis.json` and `v1_import.json`. See `v2/CUTOVER.md`. Give it the fp32 master digest from step 2.
4. **Write a small parameter file**, `smoke-params.json`. Keep the round short and pin the worker image:
   `{"n_tasks": 200, "min_scored_tasks": 50, "batch_cap": 2, "recipe": {"image_digests": ["<digest of the baked image>"]}}`
5. **First start.**
   `python -m sn120_v2 bootstrap --state-dir <state> --params smoke-params.json --stack sn120_v2.assembly:stack --genesis <dir>/genesis.json --genesis-files genesis_files.json --v1-import <dir>/v1_import.json --chain <dir>/chain.json`
   It refuses to run a second time on the same state directory.
6. **Run one round.**
   `python -m sn120_v2 run --state-dir <state> --params smoke-params.json --stack sn120_v2.assembly:stack --rounds 1 --max-idle 1`
7. **Read the result.** `python -m sn120_v2 status --state-dir <state> --params smoke-params.json`

`sn120_v2.assembly:stack` is the builder of the real parts. It lands in the foundation branch when the GPU, R2 and publisher parts are merged. Until then, steps 5 and 6 run only with `--stack sn120_v2.sim:stack` (fake parts).

Pass when all of these are true:

1. Step 6 prints one line with `"outcome"` equal to `"applied"` or `"null"`. Both are a pass: `"null"` means no LoRA beat M by more than one standard error.
2. `status` shows `"phase": "IDLE"` and `"head_folded": true`.
3. With `"applied"`: `public/current.json` in the store names the new head, and `public/rounds/<round>/batch.json` is the winner's batch.
4. Nothing is left under `pending/` in the store.
5. A second `run` on the same state directory starts with no error. This proves the journal replays.

Not a pass, and what to do:

- `"outcome": "no_decision"`: a box or store failure. Nobody is charged and the seats are back at the front of the queue. Read the error line, fix the cause, run step 6 again.
- `"outcome": "abandoned"`: the winner's files did not match their hash. The head did not change. Do not continue to cutover; report it.
- Any `refused:` line: a check failed closed. The line names the check.

## After the smoke test

The cutover is a separate step, and only Jacob starts it. The plan is in `v2/CUTOVER.md`.
