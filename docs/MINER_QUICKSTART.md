# Mine Affine SN120 v2 against the live active pool

**The live official-active scoring pool rotates every round and MUST be refetched every round.** Never treat a saved task list or the repository's static pool as the next king's draw.

The authority is fresh, cache-busted [`https://affine.io/mining.json`](https://affine.io/mining.json), specifically **`round.pool.active` and `round.pool.active_sha`**. Download `https://affine.io/public/pool/active/<sha>.json` and verify that the SHA-256 of its exact bytes equals `active_sha`. Use [`https://affine.io/public/pool/summary.json`](https://affine.io/public/pool/summary.json) **only when the round has no pool**; the summary may lag and never overrides a round pool.

```bash
# Run from the repository root: resolve the current active IDs to verified task bodies.
python v2/miner/live_pool.py --out tasks.jsonl
```

`v2/sn120_v2/envs/math_pool_v2_1.jsonl` supplies base task bodies, but is a **STALE/test-only active-set approximation and NOT a king-draw source**. The full miner reference is [`v2/miner/README.md`](../v2/miner/README.md); actual head download, training, scoring and reproduction are in [`v2/REPLICATE.md`](../v2/REPLICATE.md). Live `mining.json.params` are authoritative. All times are UTC.

## Nine-step walkthrough

1. **Register and activate.** Register an Ed25519 hotkey on SN120 (not UID 0), then make its one-time signed activation:

   ```bash
   btcli subnets register --netuid 120
   python -m sn120_v2.links activate --wallet NAME --hotkey NAME --execute
   ```

   Omit `--execute` for a preview. Never expose your seed, coldkey or mnemonic. There is **one entry per hotkey per round and no coldkey entry limit**; after a charged turn, enter again with a new hotkey/registration.

2. **Fetch the live helper's current pool.** Run `python v2/miner/live_pool.py --out tasks.jsonl`. It refreshes mutable pointers, verifies the active bytes, and reuses only verified immutable reserve-shard bytes. Refetch for the next round even if your task selection does not change.

3. **Generate rollouts and build batch JSON.** The reference rollout generator [`miner/kit/generate.py`](../miner/kit/generate.py) defaults to the live pool. Its output is rollout JSONL, not an upload file: transform it into the official batch JSON below. **Do not assume legacy `make_batch.py` supports the live-pool/current-v2 contract.**

   ```bash
   python miner/kit/generate.py --head /path/to/current-head --url http://127.0.0.1:8000 --tasks 'gen:42,math:9' --out rollouts.jsonl
   # Explicit offline/test alternative; live is the default:
   python miner/kit/generate.py --head /path/to/current-head --url http://127.0.0.1:8000 --pool local --local-pool tasks.jsonl --out rollouts.jsonl
   ```

4. **Run the ordinary check, then the live-pool check.** Use `--no-live-pool` for the offline ordinary check. Omit it to enable the default live-pool diagnostic, or use `--active FILE` for an offline saved active-list diagnostic:

   ```bash
   python v2/miner/affine_miner.py check --mining-json https://affine.io/mining.json --hotkey <ss58> --no-live-pool batch.json
   python v2/miner/affine_miner.py check --mining-json https://affine.io/mining.json --hotkey <ss58> batch.json
   python v2/miner/affine_miner.py check --mining-json https://affine.io/mining.json --hotkey <ss58> --active active.json batch.json
   ```

   Pool findings are **warning-only**. A valid training prompt need not be in the scoring pool. A file check does not train, measure KL or guarantee a crown.

5. **PUT through a freshly opened sealed link.** The opened PUT URL is a secret; do not print, save or share it. Use the publisher's current open command and the official upload helper:

   ```bash
   export AFFINE_PUT_URL="$(python -m sn120_v2.links open --hotkey <ss58> --key /private/miner.seed --links-url https://affine.io/ff/links)"
   python v2/miner/affine_miner.py upload --mining-json https://affine.io/mining.json --hotkey <ss58> batch.json
   unset AFFINE_PUT_URL
   ```

6. **Read file verdicts with the round phase.** Waiting-file checks happen while **IDLE means no round is open**, and files are checked again when a round takes entries. The validator does **not** check waiting files while a round is **RUNNING**. Mid-round verdicts describe the previous snapshot: identify it with `queue.files_checked_at`, each row's `checked_at` and `batch_sha256`, rather than assuming your latest PUT was checked.

7. **Crown only above the live threshold.** The eligible candidate with the highest delta wins, subject to the live symmetric crown gate: delta above the published tie tolerance and strict **`z > z_bar`**. The currently live **`z_bar = 0.5`**, but it can change. Read it from live params using the snippet below.

8. **Require next-round paired confirmation for payment.** The crown is applied and served immediately, but payout waits. The next round compares parent and new head with fresh paired rollouts on a shared draw. Only a successful fresh comparison, with both the recorded crown-round and fresh paired z requirements satisfied, is confirmed and paid. Crown-round winner outcomes do not replace the fresh comparison.

9. **Track confirmed kings and weights.** Only confirmed kings whose winning hotkeys remain registered split weight equally. Confirmation pays no backpay. One writer attempts weights on the hourly cadence in `params.weights_cadence_s`; a failed chain/state read preserves the last valid weights. There is no burn or owner-only fallback. Canonical winner statistics are [`https://affine.io/public/winner-stats.json`](https://affine.io/public/winner-stats.json); [`https://affine.io/winner-stats.json`](https://affine.io/winner-stats.json) serves the same bytes.

## Install the official v2 tools

Use the `v2` source, Python 3.12 and a clean virtual environment. From the repository root:

```bash
git clone https://github.com/AffineFoundation/affine
cd affine
git checkout v2
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e './v2[miner,envs,links,chain]'
```

Inspect the live miner bounds and command interfaces rather than copying caps or training knobs from old guides:

```bash
python v2/miner/affine_miner.py bounds --mining-json https://affine.io/mining.json
python v2/miner/affine_miner.py check --help
python v2/miner/live_pool.py --help
```

For GPU setup, verified head/master downloads, merged training, KL measurement and full local round scoring, use [`v2/REPLICATE.md`](../v2/REPLICATE.md). This quickstart does not replace those reproduction steps or publish fixed batch/training values.

## Resolve and verify the live pool

On every round:

1. Fetch `https://affine.io/mining.json` with a unique query parameter, no-cache headers and an explicit `User-Agent`.
2. Read `round.pool.active` and `round.pool.active_sha`. Only if the round has no pool, fall back to `https://affine.io/public/pool/summary.json`.
3. Fetch `https://affine.io/public/pool/active/<sha>.json` and require `sha256(bytes) == active_sha`. The result is a sorted, unique list of task IDs.
4. Resolve those IDs to task bodies. Refresh the manifest and other mutable pointers every time. Cache **only immutable SHA-addressed reserve shards**, verifying cached bytes before use.

**Every HTTP request in your own tooling must set `User-Agent`.** The supplied helper does so and cache-busts its requests. This stdlib example retrieves the live crown/draw parameters and authoritative pool pointer without a third-party package:

```bash
python3 - <<'PY'
import json, time, urllib.request
url = f"https://affine.io/mining.json?_={time.time_ns()}"
request = urllib.request.Request(url, headers={
    "User-Agent": "sn120-v2-miner/1",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
})
with urllib.request.urlopen(request, timeout=30) as response:
    document = json.load(response)
params = document["params"]
print("z_bar =", params["z_bar"])  # currently live: 0.5; may change
print("weights_cadence_s =", params["weights_cadence_s"])
print("irt_draw_rule =", params["irt_draw_rule"])
print("irt_middle_band =", params["irt_middle_band"])
pool = (document.get("round") or {}).get("pool") or {}
print("active =", pool.get("active"))
print("active_sha =", pool.get("active_sha"))
PY
```

### Task-body sources

- **`dapo:` / `math:`:** read bodies from the in-code [`v2/sn120_v2/envs/math_pool_v2_1.jsonl`](../v2/sn120_v2/envs/math_pool_v2_1.jsonl). Its rows include `id`, `problem`, `answer` and environment metadata. Only the live active object chooses which IDs are active; the local file is **STALE/test-only as an active list and not a king-draw source**.
- **`gen:` / reserve:** fetch the public reserve manifest [`https://affine.io/public/pool/reserve/MANIFEST.jsonl`](https://affine.io/public/pool/reserve/MANIFEST.jsonl). **`gen:` tasks are Generator-produced public reserve tasks.** Rows have `id`, `problem`, `answer` and optional public `prior_p` (Generator prior), plus environment/provenance metadata.
- For each reserve manifest row, use its `shard` value as `<name>` verbatim (including `.jsonl`) and its `sha256` as `<sha>`:

  ```text
  https://affine.io/public/pool/reserve/shards/<name>.<sha>.jsonl
  ```

  Verify each shard against its manifest SHA. Where the round supplies `reserve_manifest_sha`, verify the manifest against that too. The helper's `tasks.jsonl` is a resolved snapshot of the authoritative active IDs, not a pointer to reuse in later rounds.

## Rotation-v2 information score versus the band draw

With live `rotation_rule = "info"` (`affine-pool-rotation-v2`), each candidate's ranking score is:

```text
score = p * (1 - p)
p = 1 / (1 + exp(difficulty - head_ability))   # with sufficient history
```

Here `p` is the IRT-predicted head pass rate for measured tasks. With insufficient history, the rule uses optional public reserve `prior_p` when present, otherwise the configured `rotation_prior_p` default. Score determines active-pool rotation under the live churn rules; it is not a hard filter on training-prompt validity.

- Maximum score is **0.25 at p = 0.5**.
- **`score > 0.16` is exactly `0.2 < p < 0.8`**.
- Scores from **0.15 through 0.16 may lie outside** `(0.2, 0.8)`.

`params.irt_middle_band` belongs **only to the separate `band` draw rule**. Its preferred interval is inclusive but **not exclusive**; tasks outside retain nonzero preference. It does not govern rotation-v2 and is not an active-pool membership rule. Inspect a round's public rotation/ranking artifacts for the actual score and history, and see [`v2/POOL_ROTATION.md`](../v2/POOL_ROTATION.md) for the complete algorithm.

## Batch, checking, upload, privacy, and payout

Transform generator rollout JSONL into an official batch using a currently accepted version from `contract.batch_file.versions`:

```json
{
  "version": "affine-batch-v2",
  "hotkey": "5YourHotkeySs58Address",
  "groups": [
    {
      "prompt": "the exact training prompt",
      "rollouts": [
        {"output": "first sampled output", "reward": 1.0},
        {"output": "second sampled output", "reward": 0.0}
      ]
    }
  ]
}
```

Use finite rewards and preserve the exact prompt/output text you intend to train. Run the ordinary and warning-only live-pool checks in step 4. For full local tokenizer/reward checks, fetch the verified head using [`v2/REPLICATE.md`](../v2/REPLICATE.md), then run:

```bash
python v2/miner/affine_miner.py check --mining-json https://affine.io/mining.json --hotkey <ss58> --tokenizer /data/affine/cache --grade batch.json
```

A passing check is not a training, KL, cap-hit, crown or confirmation result. To execute the actual training/scoring path, use `python -m sn120_v2.replicate run` as documented in [`v2/REPLICATE.md`](../v2/REPLICATE.md).

### Conservative v4 SFT sizing

Start small: SFT is strong. Prefer a lower `sft_weight`, for example 0.1–0.3, and few SFT traces relative to GRPO rollouts. Measure `kl_tok` locally with `python -m sn120_v2.replicate run ...` before submission.

One anonymized, unattributed public observation: 120 SFT traces + 80x2 GRPO at `sft_weight=1.0` produced `kl_tok=0.088` and −7 percentage points, versus GRPO-only +1.7 percentage points. This is only a caution that a large SFT term at weight 1.0 can move the policy too far, not a predicted result.

The publisher lists the current sealed-link commands in `mining.json.contract.upload`. The public link file is `https://affine.io/ff/links/<hotkey>.json`; only the hotkey can decrypt it. Upload is a sealed-link `PUT` with `application/octet-stream`, as in step 5. Re-uploading replaces the stored file until your entry is frozen. Queue verdicts are phase-sensitive snapshots, so compare `checked_at` and `batch_sha256` with your bytes.

**Only crowned/applied winner batches and their training values become public. Loser, waiting and invalid batch bodies are never public.** Status metadata such as hashes, verdicts and check timestamps does not expose their prompts, outputs, rewards or training values.

After crowning, the next round's fresh paired confirmation determines payment. Confirmed registered kings divide weight equally; unregistered kings' shares are redistributed, and confirmation adds no retroactive pay. The single writer uses the hourly cadence configured in live params. Failure to read chain/state means no write and preservation of the last valid weights—never burn or owner-only fallback.

Read crowned-winner records at [`https://affine.io/public/winner-stats.json`](https://affine.io/public/winner-stats.json), or the byte-identical root alias [`https://affine.io/winner-stats.json`](https://affine.io/winner-stats.json). Every published timestamp is UTC.
