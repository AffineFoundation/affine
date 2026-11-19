# Generator miner guide

Generator is **mechanism 1 of SN120** (Bittensor netuid 120), and receives **10% of SN120 emission**. It rewards miners who find math problems that are neither always solved nor never solved by the current king. A selected task is useful training data when the king's pass rate is near the middle.

Public page: <https://affine.io/generator>

Miner kit: [`v2/miner/generator_miner.py`](https://github.com/AffineFoundation/affine/blob/bb/generator-integration/v2/miner/generator_miner.py)
Reference search: [`v2/miner/generator_search.py`](https://github.com/AffineFoundation/affine/blob/bb/generator-integration/v2/miner/generator_search.py)
Protocol: [`v2/sn120_v2/generator/`](https://github.com/AffineFoundation/affine/tree/v2/v2/sn120_v2/generator)

## 1. Install and prepare the pool

Use Python 3.11 or newer. From the repository's `v2` directory:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[generator,envs,gpu]'
```

For the kit alone, PyNaCl is required and pyarrow is optional (needed to read parquet shards and build the ID index). The reference search additionally needs vLLM 0.32.0, the pinned math-grader packages, and pyarrow. The kit imports the protocol primitives from `sn120_v2.generator`; it does not maintain a second signing implementation.

Download and verify the existing release `sn120-math-prior-v1`:

```bash
python miner/generator_miner.py pool --cache "$HOME/.cache/affine/generator" --index
```

The pool has **811,609 problems** in eight parquet shards. The release is at <https://github.com/AffineFoundation/affine/releases/tag/sn120-math-prior-v1>. The kit verifies the raw `MANIFEST.json` SHA-256:

```text
45b793b4d00d1f7225d34fd35e58018612a0ded00d96a9c23b750cfb18994115
```

It then verifies every shard's SHA-256 before reading it. IDs are the release's 20-lowercase-hex `id` column **verbatim**. Do not uppercase, renumber, hash, or convert them into another pool's ID format. The ID index contains only ID-to-shard/row locations; it is not a new dataset.

`AFFINE_GENERATOR_POOL_CACHE` can select the cache. `AFFINE_GENERATOR_POOL_MANIFEST_URL` can select a mirror of exactly the same pinned manifest bytes. These options do not permit another pool.

## 2. Activate once

Register an Ed25519 hotkey on SN120 (UID 0 cannot earn), then reuse v2's one-time activation:

```bash
python -m sn120_v2.links activate --wallet YOUR_WALLET --hotkey YOUR_HOTKEY --netuid 120 --network finney
# Inspect the dry run, then authorize the chain write yourself:
python -m sn120_v2.links activate --wallet YOUR_WALLET --hotkey YOUR_HOTKEY --netuid 120 --network finney --execute
```

An activation made for v1 remains valid. The kit itself does not register a hotkey, write chain state, or set weights. Registration, a UID other than 0, and activation are checked by the validator at the submission boundary.

## 3. Understand the UTC epoch

Epochs are four hours, aligned to **UTC midnight**. Deadlines are **00, 04, 08, 12, 16, 20 UTC**. An epoch ID is its deadline, formatted `YYYYMMDDTHHMMZ`.

The submission window for deadline `T` is **[T-4h, T)**, with the start inclusive and the deadline exclusive. For example, `20261120T0000Z` covers `[2026-11-19T20:00Z, 2026-11-20T00:00Z)`. A timestamp exactly at a deadline belongs to the new epoch. Use UTC only; never convert the schedule to a local time zone.

The king freezes at the deadline. Its head version is read from <https://affine.io/public/head.json> at that boundary. Scoring uses BF16 weights and BF16 KV cache, not FP8. The seed commitment is published at window open in `state.json`; the seed is revealed in results so the sampling can be audited.

## 4. Find candidate tasks (optional but recommended)

The reference runner uses the published king through <https://affine.io/public/read_links.json> and the existing hash-verified king fetch path. It samples the pinned pool, requests eight rollouts per task at the validator settings, grades with the production math grader, and ranks by local pass rate nearest **0.5**.

```bash
python miner/generator_search.py \
  --cache "$HOME/.cache/affine/generator" \
  --work "$HOME/.cache/affine/generator-search" \
  --sample-size 64 --count 32 \
  --ids-file ./generator-task-ids.txt \
  --report ./generator-search.jsonl
```

With no `--model-dir`, the script fetches the published folded king and starts a localhost-only vLLM server. To use an already downloaded folded base, pass `--model-dir /path/to/base --king HEAD_VERSION`. To use an existing server, pass `--url http://127.0.0.1:8100/v1` instead of `--model-dir`.

Sampling is fixed at **temperature 1.0, top_p 0.95, top_k 20, max_tokens 65,536, thinking on**. The king is the folded head with no LoRA. The local report is an estimate, not an acceptance guarantee: the validator uses the king frozen at the deadline and its own committed seeds. The ids file contains only original IDs, one per line, ready for `build`.

## 5. Open your sealed link

Fetch and decrypt `https://affine.io/generator/links/<hotkey>.json` with the hotkey's Ed25519 secret:

```bash
python miner/generator_miner.py open-link \
  --hotkey YOUR_SS58_HOTKEY --key-file "$HOME/.bittensor/wallets/YOUR_WALLET/hotkeys/YOUR_HOTKEY"
```

The command prints the **epoch, UTC expiry and upload mode**, never the presigned URL by default. The URL is a secret capability: it is read into memory, is not saved by the kit, and is not logged. `--show-url` is an explicit, normally unnecessary override. A grant expires no later than the epoch deadline; fetch a fresh link if it is too close to expiry. In the last ten minutes of an epoch, the public file may contain a separately sealed next-epoch grant. Use `open-link --epoch NEXT_EPOCH --peek` to inspect it before its UTC window opens; inspection does not authorize an early upload. Without `--peek`, a next-epoch grant is usable only once its window is open.

`--seed HEX` is supported for automation, but a hotkey file is preferable so the seed does not enter shell history. Encrypted wallets can be unlocked through `--wallet NAME --hotkey-name NAME`; that optional path uses the bittensor SDK. Never share a seed, mnemonic, presigned URL, or private submission.

## 6. Build and check a signed batch

Your task-id file must contain **at most 32 distinct IDs** from the pinned pool (at least one). Build:

```bash
python miner/generator_miner.py build \
  --hotkey YOUR_SS58_HOTKEY --key-file /path/to/hotkey.json \
  --task-ids ./generator-task-ids.txt \
  --cache "$HOME/.cache/affine/generator" \
  --output ./generator-batch.json
```

The offline check validates exact JSON schema, signature, distinct IDs, and membership in the pinned pool:

```bash
python miner/generator_miner.py check \
  --batch ./generator-batch.json --cache "$HOME/.cache/affine/generator"
```

The submission is one UTF-8 JSON object with exactly `version`, `netuid`, `mech_id`, `epoch`, `hotkey`, `pool`, `task_ids`, and `signature`. The pool field is the full release pin, including the manifest SHA-256. The Ed25519 signature is over this exact UTF-8 string, with no trailing newline:

```text
affine-generator-v1|120|1|<epoch>|<hotkey>|<pool>|sha256(newline.join(task_ids))
```

The task digest is SHA-256 of IDs joined by `\n` in submitted order, with **no trailing newline**. The complete pool string is `sn120-math-prior-v1:<raw-manifest-sha256>`. JSON canonicalization does not change this signing message.

## 7. Upload before the deadline

```bash
python miner/generator_miner.py upload \
  --batch ./generator-batch.json \
  --hotkey YOUR_SS58_HOTKEY --key-file /path/to/hotkey.json \
  --cache "$HOME/.cache/affine/generator"
```

The command rechecks the batch, verifies that its hotkey and epoch match the sealed link, then performs the presigned PUT using exactly the sealed headers. It does not print the URL. Do not manually alter signed headers or upload a batch for a different epoch.

There is **one batch per hotkey per epoch**. With a grant marked `first_upload_final`, the first upload is final and the kit never replaces it. Some grants are explicitly marked `overwrite_until_deadline`: those permit replacement before expiry, but the kit requires `--replace` to make that choice explicit. Do not treat overwrite mode as immutable. At the deadline, the validator's store-authenticated snapshot determines the eligible upload; **first upload wins** a contested task, with an exact timestamp tie broken by lower file SHA-256. IDs already accepted in any earlier epoch are ineligible.

## 8. Acceptance, ownership and payout

For a task, let `s` be passes in `n` Bernoulli rollouts and `p` its unknown king pass rate. The prior is **Beta(1,1)**, so the posterior is `Beta(s+1,n-s+1)`. Define:

```text
q = P(0.25 <= p <= 0.75)
```

Sampling proceeds in steps of **8**, up to **32**:

- Start at **8** rollouts.
- If `q >= 0.8`, accept early.
- If `q <= 0.2`, reject early.
- Otherwise sample 8 more: **8 -> 16 -> 24 -> 32**.
- At the final budget, accept when **q >= 0.5**; reject otherwise.

The boundaries are inclusive. A partial budget is final only when the scorer explicitly marks it final; the same final `q >= 0.5` rule applies then. Local search scores are only a heuristic for finding candidates.

A task can be accepted **once ever**. Accepted unique tasks enter the permanent ownership ledger. Payment is for the trailing **24 hours** (the last six four-hour epochs), equally weighted per accepted unique task:

```text
w_u = A_u / sum_v A_v
```

`A_u` is the number of payable accepted tasks credited to UID `u` in that window. Credit goes to the UID holding the earning hotkey at payout time. If there are no payable tasks, or `payout.json` is stale (older than six hours), **100% burns to UID 0**. The miner kit never sets these weights.

## 9. Privacy and commit-reveal

- Submissions are **private** while being evaluated.
- Accepted tasks, their submitter attribution, and their full accepted rollouts are **published**.
- Rejected submissions are published only as **hashes**, not their selected task contents.
- The validator publishes `sha256(secret)` at window open and reveals the secret in results. For task `task_id` and draw `j`, `seed(task,j) = int(sha256(secret:task_id:j).hexdigest()[:12], 16)`, with the secret encoded as UTF-8.
- Do not put private keys, seeds, mnemonics, presigned URLs, or private submission contents in public issues or messages.

## 10. Check status

```bash
python miner/generator_miner.py status --hotkey YOUR_SS58_HOTKEY --epochs 20261120T0000Z
# The flag is --epochs (plural), accepting one or more IDs:
python miner/generator_miner.py status --hotkey YOUR_SS58_HOTKEY --epochs 20261120T0000Z --accepted
```

This reads <https://affine.io/generator/state.json>, the epoch's <https://affine.io/generator/epochs/20261120T0000Z/results.json>, and optionally <https://affine.io/generator/accepted-all.jsonl>. Results include per-hotkey submitted/eligible/scored/accepted counts and per-task `s`, `n`, `q`, and decision once scoring is published. Before publication, a missing result is not proof of rejection.

Useful public files:

- <https://affine.io/generator>
- <https://affine.io/generator/state.json>
- <https://affine.io/generator/epochs/20261120T0000Z/manifest.json> (hashes only)
- <https://affine.io/generator/epochs/20261120T0000Z/results.json>
- <https://affine.io/generator/epochs/20261120T0000Z/accepted.jsonl>
- <https://affine.io/generator/accepted-all.jsonl>
- <https://affine.io/generator/payout.json>
- <https://affine.io/generator/generator-guide.md>
