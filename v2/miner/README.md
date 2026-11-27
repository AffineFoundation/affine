# Mine Affine SN120 v2 against the live active pool

**The official-active scoring pool rotates every round. You MUST refetch it for every round.** A saved task list, the repository's static pool, and an earlier `active_sha` do not describe a later king's draw.

The authority is a fresh, cache-busted [`https://affine.io/mining.json`](https://affine.io/mining.json): use **`round.pool.active` and `round.pool.active_sha`**. Fetch the active bytes from `https://affine.io/public/pool/active/<sha>.json` and require their SHA-256 to equal `active_sha`. Use [`https://affine.io/public/pool/summary.json`](https://affine.io/public/pool/summary.json) **only as a fallback when the round has no pool**; it can lag and never overrides a present round pool. The active object is a sorted, unique list of task IDs.

```bash
# Run from the repository root. This resolves the current IDs to verified task bodies.
python v2/miner/live_pool.py --out tasks.jsonl
```

`v2/sn120_v2/envs/math_pool_v2_1.jsonl` is a **STALE, test-only active-set approximation**. It supplies base task bodies; it is **not the live active list and is never the source of a king's draw**.

The live parameters in `mining.json.params` override this guide. Read [`docs/MINER_QUICKSTART.md`](../../docs/MINER_QUICKSTART.md) for the short workflow, and [`v2/REPLICATE.md`](../REPLICATE.md) for the actual head download, training, scoring and reproduction instructions. All published and command timestamps are UTC.

## Nine steps, from registration to payout

1. **Register and activate a hotkey.** Register an Ed25519 hotkey on SN120 (UID 0 cannot mine), then run the one-time signed activation:

   ```bash
   btcli subnets register --netuid 120
   python -m sn120_v2.links activate --wallet NAME --hotkey NAME --execute
   ```

   Omit `--execute` to preview the activation. Keep the hotkey seed, coldkey and mnemonic private. There is **one entry per hotkey per round and no per-coldkey entry limit**; a charged hotkey needs a new hotkey/registration for another turn.

2. **Fetch the live pool again.** Run `python v2/miner/live_pool.py --out tasks.jsonl` from the repository root. The helper cache-busts and verifies the live pointer and active bytes; reuse only verified immutable shard bytes. Record which `active_sha` you used, but never hardcode it into a later round.

3. **Generate rollouts and build an official batch.** The reference rollout generator, [`miner/kit/generate.py`](../../miner/kit/generate.py), defaults to `--pool live`; use `--tasks` to select live IDs. Generate from the current head and transform the generator's rollout JSONL into the official batch JSON shown below. **Generator output is not itself an upload batch. Do not assume legacy `make_batch.py` understands the live pool or current v2 contract.**

4. **Run an ordinary local check, then the live-pool check.** First use `--no-live-pool` for the ordinary offline check. Then omit it to add the default live-pool diagnostic, or use `--active FILE` with a saved, SHA-verified active object for an offline diagnostic:

   ```bash
   python v2/miner/affine_miner.py check --mining-json https://affine.io/mining.json --hotkey <ss58> --no-live-pool batch.json
   python v2/miner/affine_miner.py check --mining-json https://affine.io/mining.json --hotkey <ss58> batch.json
   python v2/miner/affine_miner.py check --mining-json https://affine.io/mining.json --hotkey <ss58> --active active.json batch.json
   ```

   Pool findings are **warnings only**, not a batch-membership admission gate. Neither command trains, measures KL, nor predicts a crown. Add the current head tokenizer and `--grade` for the full local token/reward checks; see the check section below.

5. **Open a sealed link and PUT.** The upload URL is a secret capability. Use the publisher's current `open` command, capture it without printing or committing it, and let the official helper check and upload:

   ```bash
   export AFFINE_PUT_URL="$(python -m sn120_v2.links open --hotkey <ss58> --key /private/miner.seed --links-url https://affine.io/ff/links)"
   python v2/miner/affine_miner.py upload --mining-json https://affine.io/mining.json --hotkey <ss58> batch.json
   unset AFFINE_PUT_URL
   ```

   The PUT is `application/octet-stream`. A later upload replaces your stored file until your entry is frozen.

6. **Interpret file verdicts by phase and timestamp.** Waiting files are checked while the validator is **IDLE (no round is open)**, and valid files are checked again when a round takes its entries. The validator does **not** read waiting files while a round is **RUNNING**. A mid-round verdict is the previous snapshot: match `batch_sha256` and `checked_at` to your bytes rather than treating it as a fresh check. The authoritative fields are `mining.json.phase`, `queue.files_checked_at`, and `queue.files.verdicts`.

7. **Compete for a crown.** Among otherwise eligible candidates, the highest delta wins; the current symmetric crown gate also requires delta above the published tie tolerance and **strict `z > crown bar`**. The crown bar is `params.z_crown` when that value is present and not null, and `params.z_bar` otherwise. Both are live parameters and can change. Derive them with the stdlib snippet below rather than copying a number into your tooling.

8. **Wait for fresh paired confirmation.** A crowned candidate is applied and served as the new head, but is not paid yet. In the next round, parent and new head receive fresh paired samples on a shared draw. Only a successful new comparison, with the crown-round and fresh paired z thresholds satisfied, is confirmed and paid. Old winner outcomes do not substitute for this new evidence; an unconfirmed or failed crown earns nothing.

9. **Read confirmed kings and canonical winner statistics.** Only confirmed kings whose winning hotkeys remain registered split the weight equally. Confirmation creates no back-payment. The single writer attempts weights on the hourly cadence configured by `params.weights_cadence_s`. On a failed chain/state read it writes nothing and preserves the last valid weights; there is no burn or owner-only fallback. Use [`https://affine.io/public/winner-stats.json`](https://affine.io/public/winner-stats.json); [`https://affine.io/winner-stats.json`](https://affine.io/winner-stats.json) is the same bytes at the site root.

## Install and keep the tools current

Use the official `v2` source, Python 3.12 and a clean virtual environment. From the repository root:

```bash
git clone https://github.com/AffineFoundation/affine
cd affine
git checkout v2
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e './v2[miner,envs,links,chain]'
```

The `links` and `chain` extras support sealed links and activation; `miner` and `envs` support tokenizer and official-grader checks. Serving, merged-candidate training and exact round reproduction have additional GPU dependencies. Follow [`v2/REPLICATE.md`](../REPLICATE.md), not the legacy miner training scripts, for those operations.

Useful read-only contract commands are:

```bash
python v2/miner/affine_miner.py bounds --mining-json https://affine.io/mining.json
python v2/miner/affine_miner.py check --help
python v2/miner/live_pool.py --help
```

`bounds` prints the current miner-settable fields and ranges. This guide deliberately does not freeze batch caps, training values or optimizer knobs: those belong to the live contract and its published bounds.

## Resolve the live pool without stale state

For each round, perform this sequence:

1. Fetch `https://affine.io/mining.json` with an explicit `User-Agent`, no-cache headers and a unique query parameter. Do not accept a cached contract or pool pointer.
2. Read `round.pool.active` and `round.pool.active_sha`. Only if there is no round pool, read the fallback `https://affine.io/public/pool/summary.json` instead. Never substitute the summary for a present round pool.
3. Fetch `https://affine.io/public/pool/active/<sha>.json`, where `<sha>` is the authoritative `active_sha`. Verify `sha256(downloaded_bytes) == active_sha` before parsing it. The `active` field names this same public artifact.
4. Resolve each returned ID to a body, using the sources below. Refresh the reserve manifest and other mutable pointers on every call. Cache **only immutable, SHA-addressed shard bytes**, and verify a cached shard's hash before use.

Every hand-written HTTP request must include a `User-Agent`. The supplied `live_pool.py` sends one and adds cache-busting requests; this stdlib pattern does the same for your own contract reads:

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
pool = (document.get("round") or {}).get("pool") or {}
print("active:", pool.get("active"))
print("active_sha:", pool.get("active_sha"))
PY
```

### Where task bodies come from

- **`dapo:` and `math:` IDs:** the in-code base catalog [`v2/sn120_v2/envs/math_pool_v2_1.jsonl`](../sn120_v2/envs/math_pool_v2_1.jsonl), with rows containing `id`, `problem`, `answer` and environment metadata. The catalog supplies bodies for IDs selected by the live active object. It is **STALE/test-only as an active-set list and NOT a king-draw source**.
- **`gen:` and other reserve IDs:** the public reserve manifest [`https://affine.io/public/pool/reserve/MANIFEST.jsonl`](https://affine.io/public/pool/reserve/MANIFEST.jsonl) and its SHA-addressed shards. **`gen:` tasks are Generator-produced public reserve tasks**, not private miner submissions.
- A reserve row has `id`, `problem`, `answer`, and may have a public numeric `prior_p` (Generator prior). Reserve rows also carry environment/provenance metadata. Verify the current manifest against the round's `reserve_manifest_sha` where supplied, and each shard against its manifest `sha256`.

For a manifest row whose `shard` value is `<name>` and whose `sha256` is `<sha>`, fetch:

```text
https://affine.io/public/pool/reserve/shards/<name>.<sha>.jsonl
```

`<name>` is the manifest value verbatim, including its `.jsonl` suffix. Do not strip that suffix or replace the content hash. The output of `live_pool.py` contains resolved `task_id`, `problem`, `answer` and official `prompt` fields, not a reusable active-set pointer.

The active pool governs the validator's scoring draw. It does **not** make arbitrary otherwise-valid training prompts invalid, and the warning-only pool check is not a promise that active-pool training prompts will win.

## Rotation-v2 ranking is not the band draw rule

With `rotation_rule = "info"` (`affine-pool-rotation-v2`), candidate tasks are ranked by Bernoulli Fisher information:

```text
score = p * (1 - p)
```

For a task with sufficient measured history, `p` is the IRT-predicted pass rate of the current head:

```text
p = 1 / (1 + exp(difficulty - head_ability))
```

With insufficient history, rotation uses the public reserve row's optional `prior_p` when present, otherwise the configured `rotation_prior_p` default. Rotation admits/retires candidates by this score and its live churn rules; saturated tasks are not discarded merely for having high `p`.

Useful consequences:

- The maximum score is **0.25**, at **`p = 0.5`**.
- **`score > 0.16` is exactly `0.2 < p < 0.8`**. The endpoints have score 0.16 and are not in that strict score interval.
- A score from **0.15 through 0.16 may be outside** `(0.2, 0.8)`. Histogram bins are not a hard pass-rate filter.

`params.irt_middle_band` is a **separate setting for the `band` draw rule only**. Its band is inclusive for full band preference, but **not exclusive**: outside tasks retain nonzero sampling preference. It neither limits active-set membership nor governs rotation-v2 ranking. Never claim that `irt_middle_band` controls rotation, even when its displayed edges are `[0.2, 0.8]`.

For the current rotation decision, scores, measured counts and priors, inspect the round's public `pool.rotation` and `pool.ranking` objects. For the exact algorithm and replay command, see [`v2/POOL_ROTATION.md`](../POOL_ROTATION.md).

## Derive the current crown threshold and cadence

The crown bar is `params.z_crown` when that value is present and not null, and `params.z_bar` otherwise. The next-round confirmation always uses `params.z_bar`, even when `z_crown` is set, so one comparison never stands in for the other. Read these values, `params.weights_cadence_s` and the other printed settings every time rather than treating any of them as permanent:

```bash
python3 - <<'PYSNIP'
import json, time, urllib.request
url = f"https://affine.io/mining.json?_={time.time_ns()}"
request = urllib.request.Request(url, headers={
    "User-Agent": "sn120-v2-miner/1",
    "Cache-Control": "no-cache",
})
with urllib.request.urlopen(request, timeout=30) as response:
    params = json.load(response)["params"]
z_crown = params.get("z_crown")
crown_bar = z_crown if z_crown is not None else params["z_bar"]
print("crown bar (strict z >) =", crown_bar, "from", "z_crown" if z_crown is not None else "z_bar")
print("confirmation bar (strict z >) =", params["z_bar"], "from z_bar")
print("weights_cadence_s =", params["weights_cadence_s"])
print("irt_draw_rule =", params["irt_draw_rule"])
print("irt_middle_band =", params["irt_middle_band"])
PYSNIP
```

The next-round confirmation uses fresh paired rollouts and its own comparison; winning once is not immediate payment. A confirmed king enters the paid set only at confirmation, without retroactive weight. Registered confirmed kings divide the weight equally, with shares of unregistered kings redistributed among the remaining eligible confirmed kings. If no valid write can be made, the writer preserves the last valid weights and never falls back to burn/owner weights.

## Generate, transform, and check a batch

Serve the current contract's head, then use the reference rollout generator from the repository root:

```bash
# Live is the default; this example selects particular current IDs.
python miner/kit/generate.py \
  --head /path/to/verified/current-head \
  --url http://127.0.0.1:8000 \
  --tasks 'gen:42,math:9' \
  --out rollouts.jsonl

# Explicit offline/test operation instead of the live default:
python miner/kit/generate.py \
  --head /path/to/verified/current-head \
  --url http://127.0.0.1:8000 \
  --pool local --local-pool tasks.jsonl \
  --out rollouts.jsonl
```

Without `--tasks`, live generation selects the complete fetched active list. In live mode, `--tasks` accepts exact string IDs and/or positions/ranges in that freshly fetched list. Local mode retains numeric local selection semantics. Do not let a local run silently become your live-pool provenance: refetch before the actual round.

Each generator row contains a prompt and sampled rollouts with grading metadata, not the official batch schema. Transform it to official batch JSON, assigning finite rewards and preserving the exact prompt/output text you intend to train. The legacy `miner/kit/make_batch.py` compatibility path is **not guaranteed to use the live-pool contract**; no command here promises that it does. Validate the transformed result with the official v2 checker.

An official batch has this shape (with a currently accepted `version` from `contract.batch_file.versions`):

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

Run the ordinary check and the live-pool warning check as in step 4. For token and official reward checks, fetch the verified head as described in [`v2/REPLICATE.md`](../REPLICATE.md), then run:

```bash
python v2/miner/affine_miner.py check \
  --mining-json https://affine.io/mining.json \
  --hotkey <ss58> --tokenizer /data/affine/cache --grade batch.json
```

Use `--active FILE` for an offline saved active-list diagnostic, or `--no-live-pool` to skip only that diagnostic. The live-pool check cannot make an otherwise valid file invalid merely because a prompt is outside the current scoring pool. A passing file check also does **not** mean that KL, training, cap-hit, scoring, crown or confirmation gates pass. To train and score the actual merged candidate at home, use the `fetch`, `compare` and `run` commands in [`v2/REPLICATE.md`](../REPLICATE.md).

Training targets must not reproduce other pool tasks' text or reference answers: more than `recite_max_share` of target 10-grams matching active-pool task text (own prompt excluded) blocks a crown. Per-token KL vs the head is capped at `max_kl_tok` nats/token. The configured values are in `mining.json` (`contract.duplicate_policy.recite_max_share` and `.max_kl_tok`, and the `params` block; 0 is off).

## Sealed upload, verdicts, and privacy

The current publisher exposes these commands in `mining.json.contract.upload`:

```text
activation: python -m sn120_v2.links activate --wallet NAME --hotkey NAME --execute
open:       python -m sn120_v2.links open --hotkey HOTKEY --key SEED_FILE --links-url https://affine.io/ff/links
```

The public sealed file is `https://affine.io/ff/links/<hotkey>.json`. Only your hotkey can decrypt it. Never print, save in the repository, or share the presigned PUT URL; the command substitution in step 5 passes it directly to the upload helper through `AFFINE_PUT_URL`.

The public queue is a verdict/status view, not an instant file-watching API. In IDLE it is refreshed while no round is open. During RUNNING, verdicts remain those of the previous check snapshot. Match `file_verdict`, `batch_sha256` and `checked_at`; `queue.files_checked_at` identifies the whole snapshot. Run the local official checker to diagnose your own current bytes.

**Only crowned/applied winner batches and their winner training values become public.** Loser, waiting and invalid batch bodies—including prompts, outputs, rewards and training values—are never published. Public queue/status metadata such as hashes, verdicts and check times is not publication of those batch bodies.

Canonical crowned-winner statistics are at [`https://affine.io/public/winner-stats.json`](https://affine.io/public/winner-stats.json); the root alias [`https://affine.io/winner-stats.json`](https://affine.io/winner-stats.json) serves identical bytes. The file contains crowned-winner records, not waiting or losing batches. All timestamps are UTC.
