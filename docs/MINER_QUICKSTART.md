# Mine Affine SN120: upload one batch file

Read https://affine.io/llms.txt first. The live contract is https://affine.io/mining.json
(`batch_file`, `old_format_cutoff`, `training.batch_cap`). If this page and mining.json
disagree, mining.json wins.

Since 2026-10-30 (free-file protocol) there is no mining client, no epochs, no commitments,
no TOPLOC, no log-probs, no proofs and no regrading. You upload **one JSON file** per hotkey:
prompts, your rollouts and your rewards. When your turn comes the validator copies the file,
checks its format, trains one GRPO step from the published head on it, and the promotion gate
decides GREEN or RED. Model improvement is the only gate.

Old-format (client) batches are still trained until **2026-10-31 03:00 CR (UTC-6) = 09:00 UTC**
(unix 1793437200). After that only batch files are trained.

## 1. Identity (unchanged)

Any registered SN120 hotkey (UID ≠ 0) with the signed Ed25519 Affine activation can mine.
If you have not activated yet:

```bash
python -m subnet.register --wallet YOUR_WALLET --hotkey YOUR_ED25519_HOTKEY            # preview
python -m subnet.register --wallet YOUR_WALLET --hotkey YOUR_ED25519_HOTKEY --execute  # publish
```

Your key file holds the hotkey's 32-byte Ed25519 seed as 64 hex characters (mode 0600). It is
the same key file the old client used. Never share it, a coldkey or a mnemonic.

## 2. The file: `affine-batch-v1`

```json
{
  "version": "affine-batch-v1",
  "hotkey": "5YourHotkeySs58Address",
  "groups": [
    {
      "prompt": "What is 17 * 23? Put the final answer in \\boxed{}.",
      "rollouts": [
        {"output": "<think>\n17*23 = 391\n</think>\n\n\\boxed{391}", "reward": 1.0},
        {"output": "<think>\n17*23 = 381\n</think>\n\n\\boxed{381}", "reward": 0.0}
      ]
    }
  ]
}
```

Rules (full list: mining.json → `batch_file.rules`):

- UTF-8 JSON, at most 64 MiB, exactly these keys. `hotkey` must be your hotkey.
- `prompt`: plain text, non-empty, distinct within the file. It is rendered as one user message
  with the head model's chat template and the generation prompt
  (`apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)`).
- `output`: the assistant text after that prompt (include `<think>…</think>` if you want it trained),
  tokenized with `encode(output, add_special_tokens=False)`.
- `<|im_start|>`, `<|im_end|>`, `<|endoftext|>` must not appear in prompts or outputs.
- Output ≤ 8192 tokens. Up to 8191 tokens: `<|im_end|>` (EOS) is appended. Exactly 8192: treated
  as cut at the length cap (no EOS). Prompt + output ≤ 12288 tokens.
- `reward`: any finite number, used exactly as given. Advantage = reward − mean reward of that
  prompt's rollouts (one rollout: the reward itself). The overlong penalty in `training.loss` applies.
- Any number of rollouts per prompt (≥ 1). Caps: `training.batch_cap.max_tasks_per_miner` prompts and
  `max_rollouts_per_miner` rollouts (currently 125 and 1000). Prompts beyond a cap are dropped
  whole, in file order.
- Any broken rule in the kept part makes the file invalid: you are skipped for that turn, not
  charged, and keep your queue place.

## 3. Upload

Each hotkey has one private, overwritable object. Your upload link is published sealed to your
hotkey at `https://affine.io/ff/links/<hotkey>.json` (libsodium SealedBox to your Ed25519 key
converted to X25519, re-issued every 30 minutes, each valid 24 hours). The reference helper
([`miner/affine_batch.py`](../miner/affine_batch.py), also at https://affine.io/ff/affine_batch.py)
uses only the Python standard library:

```bash
curl -O https://affine.io/ff/affine_batch.py
python3 affine_batch.py check    --hotkey 5Your... batch.json
python3 affine_batch.py upload   --hotkey 5Your... --key /private/miner.seed batch.json
python3 affine_batch.py download --hotkey 5Your... --key /private/miner.seed stored.json
python3 affine_batch.py status   --hotkey 5Your...
```

Without the helper: open the link (PyNaCl:
`SealedBox(SigningKey(seed).to_curve25519_private_key()).decrypt(base64.b64decode(sealed))`), then
`PUT` the bytes to `put_url` with `Content-Type: application/octet-stream`.

Overwrite the file as often as you like. Whatever is stored when your turn comes is what is trained.
`https://affine.io/ff/status.json` shows the validator's format check of each stored file
(valid / reason / counts, never content) within about a minute.

## 4. Turns, gate, privacy, weights

- Queue: one turn per hotkey and registration, permanently, pass or fail; order = registration
  block, then UID; UID 0 excluded. The first member with a valid file takes the next turn
  (until the cutoff also a member with a ready old-format batch). Public preview: the next 8
  (https://affine.io/training.json → `k1.queue_next`).
- Gate: the candidate must beat the head on the validator's private held-out set (z > bar_z).
  The turn batch is not excluded from the probe. The baseline is revealed only after a GREEN.
- Privacy: your file stays private until a GREEN. GREEN publishes it at
  `https://affine.io/k1/turns/<epoch>/batch.json`; RED keeps it private (only counts and its sha256).
- On GREEN the new head (bf16 + fp32 master) is published as before. Weights, flip logic and
  the nonzero weight floor are unchanged.

The old client (`subnet.miner_supervisor`) was removed from this repository.
