# Affine SN120 miner kit

Everything a miner needs to build a batch file, train the exact step the validator will train, and score it with
the exact held-out gate math, before you spend a turn. Start here. The plain-text version of this guide is
https://affine.io/llms.txt. The live contract is https://affine.io/mining.json, and it wins if the two disagree.

**What's published here.** The code below is the code the validator runs today (trainer **v8.3.4** / host-v834c, gate `prior-v1`,
free-file `affine-batch-v1/v2`). Every file's sha256 and its live source are in
[`validator/MANIFEST.sha256`](validator/MANIFEST.sha256). Only one file is changed from the live copy:
`gate_service.py`, which has the probe-pod SSH host and absolute paths replaced by placeholders. The task
selection of any open round, its seed, the prior table values, the inclusion probabilities and any private files are **not**
published. Every closed prior-v1 round reveals its secret, seed and drawn tasks; only GREEN turns reveal per-task grades.

```
miner/
  affine_batch.py               link / check / upload / download / status   (standard library only; also at affine.io/ff/affine_batch.py)
  kit/                          the tools you run
    fetch_head.py               download + sha256-verify the head (or --reference, or --tokenizer-only)
    fetch_fp32_master.py        download + verify the head's fp32 master (sha256 per file + master digest; own directory)
    generate.py                 sample rollouts from the head at the gate's settings and grade them with the gate's grader
    make_batch.py               rollouts -> affine-batch-v1/v2 file, checked with the validator's checker
    check_batch.py              the validator's FULL format check (tokenizes like the trainer)
    train_step.py               ONE v8.3.4 step with the live trainer code (KL ignored) -> candidate checkpoint (+ cand/fp32-master/)
    grad_probe.py               check your gradient against the live trainer's (same loss, same rollouts; cosine per layer block)
    gate_eval.py                the prior-v1 gate (weighted draw, fresh head, candidate, per-env Hajek estimate) on your GPUs
    gate_eval_ivw_v2.py         the old ivw-v2 gate (before 2026-11-01 17:14 CR, epoch 497 (first round r20261101T231445Z-fb048812))
    simulate_turn.sh            all of the above, end to end
    prompts.py / grade.py / sampler.py / kitpath.py   shared helpers (pool, prompt renderings, grader, vLLM client, imports)
    tests/test_kit_cpu.py       CPU self-test of the whole chain (tiny model + fake vLLM; no GPU)
  validator/                    the live code (verbatim; MANIFEST.sha256)
    trainer-v834c/              ff_trainer.py, mimo_host_trainer_v83.py (sgd_step), v8211_batched.py (loss), v83_*.py, patched/ (KL ignored)
                                trainer-config.public.json = the live engine block + TrainerConfig + runtime versions
    engine/                     subnet/mimo_grpo (the engine the host trainer drives) + vendor/mimo_verl_pinned (GRPO advantage)
    batch/ff_format.py          the batch-file parser / checker / tokenizer the lifecycle uses
    gate/                       baseline_v1.py (prior-v1 draw + estimate), prior_gate.py (round logic), gate_service.py, heldout.py, gate_core.py, gate-config.public.json
    probe/                      serve.sh (vLLM flags), probe_worker.py (sampling + grading), podctl.py, allan_src.py (lenient grader)
  data/original-math7496.tasks.json   the 7,496-task affine_math file (sha256 77a4524abc279d0e6e95ec87d0e5604f501c8e409ac5ab656ecabf060ab50fe3)
  data/gate-pool-v1.jsonl             the prior-v1 gate pool: 19,305 tasks, affine_math 7,492 + dapo 11,813 (sha256 9e05d47c4672c8e066a15079577334f0432898ea06afc7045c82862345465a8e)
  data/gate-pool-v1.manifest.json     its sources, sha256s and the 4 ids dropped for overlap with the private perf holdout
  data/gate-pool-v2.jsonl             the live gate pool: 21,574 tasks = gate-pool-v1 + 2,269 hard-v1 tasks (dapo env, dapo:11813..dapo:14081) (sha256 8d34d276bd777c2b48506687ee902041e40f68073ce71cf60f79b27471d06ebc)
  data/gate-pool-v2.manifest.json     its sources and sha256s
  requirements-trainer.txt      torch 2.14.0, transformers 5.14.1, fla 0.6.0 ... (trainer)
  requirements-gate-vllm.txt    the gate probe's pip freeze (vLLM 0.32.0, torch 2.13.0 ...)
  requirements-grader.txt       the validator-recorded eval venv freeze (math_verify 0.9.0 closure)
  requirements-sim-cpu.txt      enough to run tests/test_kit_cpu.py on a laptop
```

## 1. How a turn works

1. **Identity.** You need a registered SN120 hotkey (UID ≠ 0) with the one-time Ed25519 activation:
   `python -m subnet.register --wallet W --hotkey H --execute` (from this repo; zero fee).
2. **Queue.** You get one turn per registration (UID + registration block), whether it passes or fails.
   The order is registration block, then UID. You can see the next 8 in line at
   https://affine.io/k1.json (`queue.next`). To try again you have to re-register (pay the registration
   again) and wait for that new registration's turn. Don't deregister a hotkey that is earning.
3. **Your file.** You have one private object per hotkey that you can overwrite any time. Get your link from
   https://affine.io/ff/links/<hotkey>.json. It's a SealedBox sealed to your hotkey and re-issued every 30 minutes
   (each URL is valid for 24 h). Upload with `python3 affine_batch.py upload --hotkey <ss58> --key <seed file> batch.json`.
   Within about a minute, `affine_batch.py status` (or https://affine.io/ff/status.json) shows whether the file is valid.
4. **Your turn.** The validator walks the queue in order. The first member with a valid stored file takes the turn.
   If your file is **missing or invalid** at that moment, you're skipped: you aren't charged, you keep your place,
   and the next member is tried. The file is copied as evidence and format-checked, then the validator trains one
   GRPO step from the published head on your groups and runs the held-out gate on the candidate.
5. **GREEN** means the candidate becomes the new head (bf16 + fp32 master) and the round closes. Then published:
   - the round (`/gate/heldout/<round>.json`): secret, block B and its hash, seed, drawn tasks (id, env, problem, answer), every
     turn's decision numbers, and the GREEN turn's per-task head and candidate grades (no inclusion probabilities);
   - the decision math (Δ, SE, z, per-env numbers) in `gate.json`;
   - the winning file (`batch.json`) with its lr / kl (`turn.json`).

   **RED** means the head is kept and only the decision, z / gain, the counts and the file's sha256 are published.
   Your file, your lr/kl and your probe results stay private. While a round is open only its commitments (secret, then
   block B, then the draw) are public. When a round closes, for any reason, its secret, seed and drawn tasks are revealed;
   per-task grades only for a GREEN turn. RED candidates' probe results are never published. See https://affine.io/llms.txt section 7.
6. **Winning = weight.** The last (up to) 10 distinct GREEN UIDs each get 1/N of the miner weight, for as long as
   the hotkey that earned it stays registered on that UID. Unearned weight burns to UID 0, the weights are never
   all zero, and they're set hourly. Current payees are listed at https://affine.io/k1.json (`payees`).

## 2. The batch file

```json
{
  "version": "affine-batch-v2",
  "hotkey": "5YourHotkeySs58Address",
  "train": {"lr": 5e-7, "kl": 0.02},
  "groups": [
    {"prompt": "Solve the math problem. Reason step by step in plain text, then end your response with the final answer in `\\boxed{}`. Emit exactly one `\\boxed{}` block and nothing after it.\n\nWhat is $17\\cdot 23$?",
     "rollouts": [
       {"output": "<think>\n17*23 = 391\n</think>\n\n\\boxed{391}", "reward": 1.0},
       {"output": "<think>\n17*23 = 381\n</think>\n\n\\boxed{381}", "reward": 0.0}]}
  ]
}
```

- **v1** has exactly `version, hotkey, groups`. **v2** adds an optional `train` with `lr` in [1.25e-7, 1e-6]
  (default 5e-7) and `kl` in [0.005, 0.1] (default 0.02; **accepted but ignored** as of trainer v8.3.4 / 2026-11-05 — keep sending it so old files still validate). From epoch 575 (trainer v8.3.3l) also `step_mult` and the optional
  knobs below; every one defaults to the previous trainer, so a file without them trains bit-identically to before.
  The bounds are inclusive and values are never clamped (a non-integer step_mult such as 2.0 is invalid).

| field | default | range | what it does |
|---|---|---|---|
| `train.step_mult` | 1 | integer: 1, 2 | multiplies the whole one-step update: lr and the per-tensor cap scale together, one gradient, no extra trainer time |
| `train.lr_scale_attn` | 1 | 0 .. 2 | multiplies the step of every attention tensor (self_attn.*, linear_attn.*); 0 freezes them |
| `train.lr_scale_mlp` | 1 | 0 .. 2 | multiplies the step of every mlp.* tensor; 0 freezes them |
| `train.lr_scale_embed` | 1 | 0 .. 2 | multiplies the step of embed_tokens; 0 freezes it |
| `train.lr_scale_lm_head` | 1 | 0 .. 2 | multiplies the step of lm_head; 0 freezes it |
| `train.lr_scale_early` | 1 | 0 .. 2 | multiplies the step of every tensor in decoder layers 0..15 (on top of the group scale); 0 freezes them |
| `train.lr_scale_late` | 1 | 0 .. 2 | multiplies the step of every tensor in decoder layers 16..31 (on top of the group scale); 0 freezes them |
| `train.adam_eps` | 1e-8 | 1e-10 .. 1e-3 | eps in the normalized-gradient term g / (abs(g) + eps); larger eps shrinks the per-tensor step of tensors with small gradients (the step direction stays g / norm(g)) |
| `train.weight_decay` | 0.01 | 0 .. 0.05 | decoupled weight decay inside the per-tensor step norm (2-D non-bias tensors); it changes step sizes, not the direction |
| `train.clamp_mode` | per_tensor | per_tensor, global_norm or both | per_tensor: each tensor's step is capped at c_t = 2e-5 x norm(W_t) (today). global_norm: no per-tensor cap; the whole step is scaled down to global_norm_cap if larger. both: per-tensor cap, then the global cap at global_norm_cap |
| `train.global_norm_cap` | 1 | 0.005 .. 1 | the global step-norm cap used by clamp_mode global_norm / both (ignored with per_tensor, which keeps the fixed cap 1.0) |
| `train.loss_agg` | seq_mean | seq_mean or token_mean | seq_mean: every rollout weighs the same (mean over its tokens, then over rollouts; today). token_mean: every token weighs the same (long rollouts weigh more) |
| `train.adv_center` | mean | mean, none or group_norm | advantage of a rollout: mean = reward - mean reward of its prompt group (today); none = the raw reward; group_norm = (reward - mean) / (std + 1e-6) within the group |
| `train.adv_clip` | none (no clip) | 0.1 .. 100 | if set, advantages are clipped to [-adv_clip, +adv_clip] after centering; omit it for no clip (today) |

  A value out of range, a non-number or an extra key makes the file invalid (you're skipped, not charged).
  Putting `train` in a v1 file is also invalid.
- The file is UTF-8 JSON, at most 64 MiB. Prompts must be non-empty and distinct, and each needs at least one rollout.
- **prompt.** It's rendered as ONE user message: `apply_chat_template([{"role":"user","content":prompt}], add_generation_prompt=True)`
  with the head tokenizer. There's no system message, and the generation prompt is `<|im_start|>assistant\n`.
- **output.** It's encoded with `encode(output, add_special_tokens=False)`. `<|im_start|>`, `<|im_end|>` and
  `<|endoftext|>` (ids 248044–248046) are forbidden. Up to 12,287 tokens, `<|im_end|>` is appended (it ended).
  Exactly 32,768 tokens counts as capped (no EOS) for training/batch. More than 32,768 is invalid in a batch file. Prompt + output can be at most 36,864 tokens for training.
  **32,768 output tokens apply to training** (https://affine.io/caps.json): generation (sample with `max_tokens` 32,768), batch files and training. The promotion gate/probe remain at 65,536 output / 69,632 context.
  The one source of training cap numbers is
  [https://affine.io/caps.json](https://affine.io/caps.json) (repo `caps.json`); this README, llms.txt, mining.json, the kit,
  the batch validator, the trainer and the gate are checked against it by `tools/caps/caps_check.py`.
- **Caps.** At most 125 prompts and 1,000 rollouts (live value: mining.json `training.batch_cap`). Prompts past
  a cap are dropped whole, in file order.
- **reward.** Any finite number, used as given. Advantage = reward − the mean reward of that prompt's
  rollouts (no std normalisation), so **a prompt with one rollout gets advantage = its reward** (+1 for a correct
  reward-1 rollout). The trainer adds an overlong penalty of −0.5·clip((len−26923)/(32768−26923), 0, 1) to the reward
  (len incl. EOS; window = caps.json `overlong_soft_start`..`overlong_max_tokens`).
- Check a file the way the validator does: `python3 kit/check_batch.py --hotkey <ss58> --tokenizer ./head batch.json`
  (`affine_batch.py check` does the structural part without a tokenizer).

## 3. The training step (v8.3.4): what `kit/train_step.py` runs

This is one stateless step from the published head. All the code is in `validator/trainer-v834c` and `validator/engine`.
The live settings are in `trainer-config.public.json`.

- **Loss** (`v8211_batched.train_update_batched`): GRPO with one group per prompt, PPO clip 0.2, dual clip 3.0.
  As of trainer **v8.3.4** (2026-11-05) the KL term is **off**: `train.kl` is still accepted and parsed in [0.005, 0.1]
  so existing files keep validating, but the update forces `kl_coefficient=0` and does **not** load the Genesis reference
  (`87de8304…`). It uses a derail mask (window 32, threshold −3, tail 64) and aggregates `seq-mean-token-mean` × 0.5635:
  L = 0.5635·(1/N)·Σ_i (1/K_i)·Σ_t [pg]. Gradients are accumulated in fp32 and globally clipped at 1.0. Every log-prob
  is recomputed from the head (temperature 1). The trainable tensors are `model.language_model.*` and `lm_head.*`.
- **Support renorm.** The PG term uses log q of your token renormalised over the head's own top-20 → top-p 0.95
  support. **A sampled token outside that support is dropped from the PG term only.** It still counts in
  the derail mask (KL term is off), and it's only logged (`support_renorm` in turn.json). Files are never rejected for it. The practical
  consequence: text the head would not sample (hand-written, or from another model) trains mostly nothing (KL is off).
  e368 had 0.14% of its tokens outside the support.
- **Step** (`mimo_host_trainer_v83.sgd_step`: fresh-adam, fp32 master, direction normgrad). For each tensor t, with
  W_t = the fp32 master:
  d_t = 5e-7·(g_t/(|g_t|+1e-8) + 0.01·W_t·[W_t ≥ 2-D and not a bias]);
  per-tensor cap c_t = 2e-5·(1−1e-6)·‖W_t‖; n0_t = min(‖d_t‖, c_t); with m = lr/5e-7, n_t = min(m·n0_t, c_t);
  the global cap is s = min(1, 1/‖n‖); Δ_t = s·n_t·g_t/‖g_t‖; M_C = M_H − Δ; and the candidate = bf16_rne(M_C).
  With the default lr, many tensors already sit at the per-tensor cap, so lr 1e-6 moves the model less than 2×
  (at e368, 242 of 427 tensors were at the cap; turn.json `training.train_params.step`).
  **step_mult k** (v8.3.3l, from epoch 575; proposed in #120 by duongna): n_t := k·n_t before the global cap, i.e. lr and the
  per-tensor cap scale together and every tensor moves k× (the capped ones too). One gradient, one step, no extra trainer
  time. The default step norm is ≈0.023, so the global cap (1.0) does not fire for any allowed k. k = 1 is exactly the v8.3.3j step.
  **Knobs** (v8.3.3l, only when one differs from its default): eps = adam_eps and 0.01 = weight_decay in d_t; n_t is multiplied
  by the tensor's group scale (lr_scale_attn / mlp / embed / lm_head; norms 1) × lr_scale_early (layers 0–15) or lr_scale_late
  (16–31); clamp_mode global_norm drops c_t and caps the whole step at global_norm_cap (both: c_t and then global_norm_cap);
  adv_center / adv_clip set the GRPO advantages; loss_agg token_mean weighs tokens instead of rollouts. The direction is
  always g_t/‖g_t‖. Code: `validator/trainer-v834c/mimo_host_trainer_v83.py` (`knob_scale`, `_fresh_adam_step`) and
  `v8211_batched.py` (`_apply_knobs`); `kit/train_step.py` applies them exactly like the live wrapper.
- **Kernels.** On a GPU `train_step.py` runs the live trainer's own training forward: engine `packed` with the trainer's
  kernels installed on the actor (KL reference no longer loaded as of v8.3.4) (`validator/trainer-v834c/v83_fast_kernels.py`: fla 0.6.0
  `causal_conv1d` + `chunk_gated_delta_rule` varlen, torch gated RMSNorm, FlashAttention-2 varlen; packed log-probs as in
  `v83_dp.py`). What still differs from the live run: one process instead of 8 GPUs with an NCCL sum, and the micro-batch
  grouping, so the gradient differs at the bf16 rounding level (sum order, atomics), not bit-exactly.
  Until 2026-11-05 the kit ran engine `padded` with the stock transformers forward, which is not the trainer's math: the
  GatedDeltaNet short conv falls back to torch `F.conv1d` (bf16 backward), and the attention / gated-norm kernels differ.
  Allan (catoneone) measured that path at cosine 0.58 to an fp32 reference and a 1.36× norm, layers 0–7 mostly noise, on the
  E718 file with head 58bcc262 ([sn120-k1-fixes exp024/phase5](https://github.com/catoneone/sn120-k1-fixes/tree/main/exp024/phase5)).
  On our 1×H200 re-run (head 7214d10a, E745 and E718 files) the old path was closer (cosine 0.957–0.967 to the trainer, layers
  0–7 0.91–0.94) and the current default 0.9997–0.9998 in every block, so the size of the error depends on the setup:
  measure yours with `grad_probe.py`. `--engine padded` is kept only for the CPU self-test.
- **The fp32 master.** The live trainer steps the head's fp32 master M_H, not fp32(bf16 head). Every GREEN publishes it
  next to the bf16 head: https://affine.io/k1/fp32-master/<head>.json (files + sha256, master_digest, presigned download
  links in `download`; current head: /k1/fp32-master/current.json). `kit/fetch_fp32_master.py` downloads it (~36 GB,
  9 float32 shards, weights only) and verifies every file's sha256 and the master digest (= the GREEN turn.json
  `training.optimizer.sgd.fp32_master.output_master_digest`). `train_step.py` uses `HEAD_DIR-fp32-master` automatically
  when it was verified for that head (or `--master DIR`; `--no-master` forces the fp32(bf16) bootstrap, which drifts from
  the gate's candidate). Keep the master in its own directory: vLLM loads every `*.safetensors` in a model directory.
  The candidate's own master is written to `cand/fp32-master/` for the same reason.
- **One GPU.** `--activation-checkpointing` (opt-in, off by default) recomputes each decoder layer in the backward pass
  instead of storing its activations: same values (recomputed), slower, much less memory. Use it when a full file OOMs on one GPU (1x H200).

```
python3 kit/fetch_head.py --out head   # --reference unused as of v8.3.4 (KL off)
python3 kit/fetch_fp32_master.py --out head-fp32-master        # fp32 master of the same head, sha256 + master digest verified
python3 kit/train_step.py --head head --ref ref --batch batch.json --hotkey <ss58> --out cand --report step.json \
    [--activation-checkpointing]                                # uses head-fp32-master automatically
```

### Check your gradients (`kit/grad_probe.py`)

`grad_probe.py` computes the gradient of one fixed loss on the same rollouts with different implementations and compares
them per tensor (cosine and norm ratio, overall and per 8-layer block). Any public GREEN batch works
(`https://affine.io/k1/turns/<epoch>/batch.json`, hotkey in that turn's `turn.json`); one H200, a few minutes per run.

```
python3 kit/grad_probe.py run --head head --batch batch.json --hotkey <ss58> --mode kit     --out g_kit.safetensors
python3 kit/grad_probe.py run --head head --batch batch.json --hotkey <ss58> --mode trainer --out g_trainer.safetensors
python3 kit/grad_probe.py compare g_kit.safetensors g_trainer.safetensors
```

`kit` is what `train_step.py` runs; `trainer` is the live trainer's forward set up directly from `validator/trainer-v834c`;
`legacy` is the old `padded` path; `fp32ref` is a slow fp32 pure-torch reference. Kit vs trainer should give a cosine of at
least 0.99 and a norm ratio near 1.0 in every block, layers 0–7 included. Much lower means your environment (package
versions, kernels, edited files) does not compute the gate trainer's gradient.
Our 1×H200 numbers (24 rollouts, head 7214d10a, E745 file): kit vs trainer cosine 0.9998, norm ratio 1.0002 (layers 0–7:
0.9997, 1.0006); legacy vs trainer 0.957, 0.993 (layers 0–7: 0.910); every bf16 path vs fp32ref about 0.96.

## 4. The gate (held-out, prior-v1 since 2026-11-01 17:14 CR, epoch 497 (first round r20261101T231445Z-fb048812)): what `kit/gate_eval.py` runs

- **Pool:** `data/gate-pool-v2.jsonl` (from the flag change of 2026-11-05 13:43 CR (epoch 767): the first round opened on the next new head; before: `data/gate-pool-v1.jsonl`, 19,305 tasks), 21,574 public
  tasks: affine_math 7,492 (MATH; `math:<index>` into `data/original-math7496.tasks.json`, minus 4 near-duplicates of the
  private perf holdout) + dapo 14,082 = DAPO-Math-17k 11,813 (integer answers, built by `priors/dapo/build_dapo.py`) + the
  hard-v1 slice, 2,269 harder problems (dapo:11813..dapo:14081; CrystalMath-rewritten + INTELLECT-3-RL math, decontaminated, kept at a 5-60% solve
  rate of king 8389b90b at 32K, see `priors/hard_v1/`). The pool is public; only the per-round draw is secret. Manifest with
  sha256s: `data/gate-pool-v2.manifest.json`. The gate
  prompt for both envs is `[system: MANDATE, user: problem]` with `enable_thinking=True`, where MANDATE =
  "Solve the math problem. Reason step by step in plain text, then end your response with the final answer in
  `\boxed{}`. Emit exactly one `\boxed{}` block and nothing after it."
- **Sampling:** temperature 1.0, top_p 0.95, top_k 20, 65,536 output tokens (the task cap), 69,632 context, vLLM 0.32.0 on
  8×B200. A round's
  cap is `mining.json` → `training.promotion_gate.heldout.current_round.settings.max_tokens`, the same for head and candidates
  (`probe/serve.sh`: `--dtype bfloat16 --max-model-len 36864 --enable-prefix-caching --max-num-seqs 512`). One request per
  sample, seed = int(sha256(`<plan seed>:affine_gatepool:<probe index>:j`)[:12], 16); probe index = the math index, or
  7496 + j for `dapo:j`.
- **Grader** (`kit/grade.py` = `probe_worker.grade`, the same for both envs, head and candidates): a capped reply is neutral
  (unresolved). Otherwise the final channel after one `<think>…</think>` must hold exactly one balanced `\boxed{}`, checked
  with math-verify 0.9.0 against `\boxed{gold}`; if that fails, the lenient fixes allan-lenient-v1 (P1/P2 projection, E1–E7
  equivalence; `probe/allan_src.py`) are tried. A grader crash or a grade over 300 s counts as not passed.
- **The prior (method public, values private).** A separate sampler pod keeps sampling the current head on the whole pool
  (k = 4 per task per pass) and keeps per-task s/n = solved/sampled (`priors/`). On every head change all counts are multiplied
  by 0.5. A task without samples starts at its env's average. Posterior p ~ Beta(s+1, n−s+1).
- **A round (one per head).** At open the gate snapshots the prior table (its sha256 is published) and computes
  π_i ∝ √E[p_i(1−p_i)], mixed with a 10 % uniform floor, capped at 0.10, scaled to an expected N = 500
  (`baseline_v1.inclusion`). Seed = commit-reveal: the commitment sha256(version|secret|secret) is published at open; after the
  upload deadline the gate publishes B = finalized block + 3; once B is final, seed = sha256(version|seed|secret|B|hash(B)).
  Task i is drawn iff u_i < π_i, u_i = sha256(version|draw|seed|id)/2^64 (Poisson sampling, `baseline_v1.draw`). The head runs
  3 blocks × 4 = 12 samples on every drawn task; each candidate runs k_cand = 8 samples per drawn task with independent seeds
  (`baseline_v1.cand_seed`). The turn batch is not excluded. A round ends on a GREEN, a head change or 8 decided turns; then
  everything except π is revealed (`/gate/heldout/<round>.json`: secret, B, hash, seed, drawn tasks, every turn's numbers, the
  GREEN turn's per-task grades).
- **Score** (`baseline_v1.estimate`): per env, the Hájek mean m_e = Σ(d_i/π_i)/Σ(1/π_i), d_i = candidate rate − head rate;
  Δ = Σ_e share_e·m_e with share = env size / pool size (affine_math 0.388, dapo 0.612). SE = √(Σ_e share_e²·Var_e + σ_run²),
  Var_e = the Hájek sampling + rollout variance, σ_run = 0: no run-noise floor in rounds opened after the 2026-11-03 change
  (0.003 before; the 2026-11-01 head-vs-itself test found no measurable run-to-run noise). z = Δ/SE.
- **GREEN iff** Δ > 0 **and** z > 1 (strict) **and** cap-hit < 0.33 **and** ≥ 100 drawn tasks scored (≥ 95 % of the draw).
  Cap-hit (task-level, from E507): the share of all the candidate's answers on the scored baseline tasks that hit the
  round's token cap (currently 65,536), every task counted alike: **RED if ≥ 33% of the candidate's baseline answers hit the token cap** (before
  E507: affine_math cap-hit < 0.20).
- **z bar: fixed at 1.00** (https://affine.io/gate.json `z_bar`, mining.json `heldout.z_bar`). The bar each turn was judged
  against: `entries[].thresholds.z_min`. History: from E553 the bar is fixed at z > 1; E513–E552 used a time-relaxed bar.
  The bar (= SE) is expected near +0.8 to +1.2 points of pool-wide gain.
- **What you can verify:** with `priors/gate/verify_round.py <round.json> data/gate-pool-v2.jsonl (v1 for rounds before the switch)`: the secret commitment, the block
  hash on chain, the seed, the draw commitment and that every drawn task is in the pool; plus the GREEN turn's grades. **What you
  can't:** recompute Δ/SE/z exactly or check u_i < π_i, because π comes from the private prior table (only its snapshot sha256 is
  published).

`gate_eval.py` runs this procedure with its own local seed. With no `--prior` every task gets its env average, so the draw is close
to uniform; pass `--prior my.jsonl` (`{"id","s","n"}` per task from your own head samples) to mimic the live weighting. Settings come
from `validator/gate/gate-config.public.json` (`heldout.prior_v1`). Your GPUs and vLLM build aren't the gate's, so expect small
differences. `--baseline` reuses a head baseline across candidates. The old ivw-v2 procedure is `kit/gate_eval_ivw_v2.py`.

## 5. End to end

```
pip install -r miner/requirements-trainer.txt vllm==0.32.0 math-verify==0.9.0 httpx huggingface_hub
HOTKEY=5Your... bash miner/kit/simulate_turn.sh ./turn          # head (+ fp32 master) -> rollouts -> batch -> step -> gate
#   ACT_CKPT=1 for one GPU (activation checkpointing); FP32_MASTER=0 to skip the 36 GB master download
python3 miner/affine_batch.py upload --hotkey 5Your... --key /private/miner.seed ./turn/batch.json
```

CPU self-test (no GPU, about 3 minutes):

```
pip install -r miner/requirements-sim-cpu.txt
python3 miner/kit/fetch_head.py --out /tmp/tok --tokenizer-only
python3 miner/kit/tests/test_kit_cpu.py --tokenizer /tmp/tok
```

## 6. How to win: what has worked so far (observations, not guarantees)

- **The first GREENs under the free-file rules** (e354 z 3.88, e356 z 1.03, e358 z 2.96, e360 z 2.42, e368 z 1.70, all on
  2026-10-31) were each a **v1 file with the default lr and kl, 125 prompts × 1 correct rollout**. The prompts were MANDATE + "\n\n" +
  a pool problem (125/125 from the pool, mostly Level 4–5). Each rollout was the head's own sample with a
  `<think>` block and exactly one final `\boxed{}`. Since e342 almost every turn has had the same public counts
  (125 prompts × 1 correct rollout), including the REDs e362–e367 and e370–e376. Several of those REDs scored strongly
  negative (z −3 to −5.6 at e363, e365–e367, e370, e371). So the shape is common, but it isn't sufficient: which problems
  you pick and which samples you keep decide the sign.
- **Borderline difficulty.** The gate only scores tasks the head sometimes solves (0 < c < 4). Correct samples on
  problems the head usually fails or caps out on carry the signal. `make_batch.py --select correct1` picks those
  (hardest first, shortest correct sample).
- **Fewer capped replies.** In the performance eval (full affine_math pool) the pass rate went from
  0.8448 (E0) to 0.9014 (E368), while the cap-hit rate fell from 0.1175 to 0.0736. About 4.4 of those 5.7 points
  come from fewer replies hitting the token cap, and the mean output length dropped from 2,043 to 1,769 tokens. Short correct
  rollouts (well under the overlong window, where the overlong penalty starts) are consistent with that.
- **Sample from the head** with top_k 20 / top_p 0.95 / temperature 1. Off-support tokens are dropped from the PG term.
- The 1,000-rollout files with mixed outcomes (e339–e341, e343; z 0.70, −0.78, 0.44, −0.80) went RED, though those ran under the
  older sign step (before e354).
- **v2 `train` values have won too:** E537 (lr 1.25e-7, kl 0.02 given) and E485, E544, E552, E555, E557 (kl 0.005, default lr) were v2 files.
  Every GREEN file is public byte-exact at https://affine.io/k1/turns/<epoch>/batch.json, with its lr / kl in `turn.json`.
- At the default lr the per-tensor cap dominates: at e368, 242 of 427 tensors were at the cap, and they carried 98.8% of the
  step norm (`train_params.step.capped_norm_fraction_default`). Raising lr barely enlarges the step, while lowering it shrinks every tensor.
- Simulate before you spend your registration: `gate_eval.py` uses the same math. A local z near 1 is a coin flip.

## 7. Public endpoints

| what | where |
| --- | --- |
| contract (format, caps, queue, link) | https://affine.io/mining.json |
| this guide as text | https://affine.io/llms.txt |
| training parameters, support renorm, queue | https://affine.io/training.json |
| queue, payees, weights | https://affine.io/k1.json |
| turns (GREEN: batch.json, turn.json) | https://affine.io/k1/turns/index.json, /k1/turns/<epoch>/turn.json |
| everything about a GREEN (baseline, probe, tasks, math, file, lr/kl) | /k1/turns/<epoch>/green.json, /k1/turns/<epoch>/tasks.json |
| gate decisions / open round / reveals | https://affine.io/gate.json, /gate/heldout/current.json, /gate/reveal/<epoch>.json |
| your file's format check | https://affine.io/ff/status.json |
| step recipe / fp32 masters / replay | https://affine.io/k1/spec/step-recipe.json, /k1/fp32-master/<head>.json, /k1/replay/README.md |

## 8. Known gaps

- **fp32 master history.** Download links are kept for the newest 3 GREEN heads (older manifests keep their sha256s).
  The e651 head 4df7fb61 has no published master (its store entry was gone before the upload).
- **Trainer pod pip freeze.** Only the versions the trainer reports are published (torch, transformers, toploc, fla).
  `requirements-trainer.txt` pins those, plus what the code imports.
- **8-GPU reduction.** The local step runs the trainer's packed kernels in one process; the live run sums 8 GPUs
  with NCCL, so gradients match at the bf16 rounding level, not bit-exactly (`kit/grad_probe.py` measures it).

## Audit the eval runtime before registering

See [the pinned live-probe recipe](validator/probe/runtime/MINER-REPRO.md) and run
`python3 miner/kit/verify_eval_runtime.py`. Parallel evaluation is live from epoch 843 (2026-11-07): each candidate is
judged on its own 8-GPU eval node with exactly this pinned runtime and config (bf16, no fp8, max-model-len 69632,
max-num-seqs 512) and the gate's own draw, seeds, estimator and thresholds; only the hardware differs. Phase 1 applies
fresh GREENs only (see llms.txt section 6 and ops/parallel_eval/live/). The audit itself is a file/index check,
not a certified end-to-end GPU replay.

The batch top-level `version` must be exactly `affine-batch-v1` or
`affine-batch-v2` (not a trainer/kit release number).
