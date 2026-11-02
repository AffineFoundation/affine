# Mine Affine SN120: upload one batch file

Read https://affine.io/llms.txt first. The live contract is https://affine.io/mining.json
(`batch_file`, `training.batch_cap`). If this page and mining.json
disagree, mining.json wins.

Batch files are the only mining format (free-file protocol). No mining software is needed and
your rewards are used as given. You upload **one JSON file** per hotkey:
prompts, your rollouts and your rewards. When your turn comes the validator copies the file,
checks its format, trains one GRPO step from the published head on it, and the promotion gate
decides GREEN or RED. Model improvement is the only gate.

## 1. Identity (unchanged)

Any registered SN120 hotkey (UID ≠ 0) with the signed Ed25519 Affine activation can mine.
If you have not activated yet:

```bash
python -m subnet.register --wallet YOUR_WALLET --hotkey YOUR_ED25519_HOTKEY            # preview
python -m subnet.register --wallet YOUR_WALLET --hotkey YOUR_ED25519_HOTKEY --execute  # publish
```

Your key file holds the hotkey's 32-byte Ed25519 seed as 64 hex characters (mode 0600). Never share it, a coldkey or a mnemonic.

## 2. The file: `affine-batch-v1` (or `affine-batch-v2` with optional lr / kl)

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

### Optional: `affine-batch-v2`: optional lr and kl (from 2026-10-31)

A v2 file is a v1 file with `"version": "affine-batch-v2"` and an optional top-level `train`:

```json
{
  "version": "affine-batch-v2",
  "hotkey": "5YourHotkeySs58Address",
  "train": {"lr": 8e-7, "kl": 0.01},
  "groups": [ ...same as v1... ]
}
```

| field | default | allowed (inclusive) | what it does |
| --- | --- | --- | --- |
| `train.lr` | 5e-7 | 1.25e-7 .. 1e-6 | step size of your turn's one step |
| `train.kl` | 0.02 | 0.005 .. 0.1 | kl_coefficient of your turn's loss |

- Both fields are optional; an omitted field takes its default. `train` itself is optional in v2.
  v1 files keep working unchanged (they train with the defaults). `train` in a v1 file is invalid.
- Out of range, not a number (string, true/false, null) or any other key under `train` = the whole
  file is invalid (skipped, not charged, you keep your place). Values are never clamped.
- How lr enters (trainer v8.3.3j): m = lr / 5e-7. For each tensor, n0_t = the default step norm
  (min(||d_t||, c_t), c_t = 2e-5 x ||W_t|| the per-tensor cap); your step norm is n_t = min(m x n0_t, c_t),
  then the global cap 1.0 and the direction g_t/||g_t|| as before: Delta_t = s x n_t x g_t / ||g_t||.
  Below the default every tensor's step shrinks by exactly m. Above it only tensors still under the
  per-tensor cap grow, up to the cap (the cap stays a ceiling; with the default many tensors already
  sit at it, so lr 1e-6 moves the model less than 2x).
- How kl enters: it replaces kl_coefficient in
  L = loss_scale x (1/N) x sum_i (1/K_i) x sum_t [pg + kl x k3] for your turn only.
- Everything else is fixed: per-tensor cap, global cap and gradient clip, overlong penalty, starting
  head, one step per turn, batch cap (125 x 8), queue and gate.
- Your effective lr / kl are published with your turn only if it is GREEN (turn.json
  training.train_params, mining.json training.miner_train_params). RED: private like the file.
- Check locally: python3 affine_batch.py check --hotkey <ss58> batch.json (reports format, lr, kl).

## 3. Upload

Short version: write `batch.json` → `python3 affine_batch.py check …` → `python3 affine_batch.py upload …`
→ within about a minute `python3 affine_batch.py status …` (or https://affine.io/ff/status.json) shows
`valid: true`. Re-upload whenever you like.

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
  block, then UID; UID 0 excluded. The first member with a valid file takes the next turn.
  Public preview: the next 8
  (https://affine.io/training.json → `k1.queue_next`).
- Gate: head and candidate are scored on the same tasks of the validator's private held-out set
  (4 seeded samples per task, same seeds on both sides). GREEN needs a positive weighted per-task gain
  (inverse-variance-weighted mean of candidate − head pass-rate differences) with z > 1.00 (strict; the bar is fixed at 1.00, https://affine.io/gate.json `z_bar`),
  fewer than 33% of the candidate's baseline answers hitting the token cap (task-level, from E507; before: under 0.20) and at least 100 kept tasks. The turn batch is not excluded from the
  probe (from E296). The baseline is revealed only after a GREEN or when the round closes. Exact text:
  https://affine.io/gate.json → `rule`.
- Fresh turns (Jacob, 2026-10-30): the 10 hotkeys whose turn since the held-out gate went live was RED with
  0 < weighted_z ≤ 1.00 each get one fresh turn ahead of the line, oldest first, under the batch-file rules.
  It stays open until it is actually used (a missing or invalid file is a skip). List:
  https://affine.io/training.json → `k1.queue_grants`.
- Privacy: your file stays private until a GREEN. GREEN publishes it at
  `https://affine.io/k1/turns/<epoch>/batch.json`; RED keeps it private (only counts and its sha256).
- On GREEN the new head (bf16 + fp32 master) is published as before. Weights, flip logic and
  the nonzero weight floor are unchanged.
