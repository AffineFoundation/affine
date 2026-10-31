# Affine SN120 miner kit

Everything a miner needs to build a batch file, train the exact step the validator will train, and score it with
the exact held-out gate math, before you spend a turn. Start here. The plain-text version of this guide is
https://affine.io/llms.txt. The live contract is https://affine.io/mining.json, and it wins if the two disagree.

**What's published here.** The code below is the code the validator runs today (trainer v8.3.3j, gate `ivw-v2`,
free-file `affine-batch-v1/v2`). Every file's sha256 and its live source are in
[`validator/MANIFEST.sha256`](validator/MANIFEST.sha256). Only one file is changed from the live copy:
`gate_service.py`, which has the probe-pod SSH host and absolute paths replaced by placeholders. The task
selection of any open round, its seed, and any private files are **not** published. A round's seed and task
ids are only revealed if a GREEN closes it, and RED turns are never revealed.

```
miner/
  affine_batch.py               link / check / upload / download / status   (standard library only; also at affine.io/ff/affine_batch.py)
  kit/                          the tools you run
    fetch_head.py               download + sha256-verify the head (or --reference, or --tokenizer-only)
    generate.py                 sample rollouts from the head at the gate's settings and grade them with the gate's grader
    make_batch.py               rollouts -> affine-batch-v1/v2 file, checked with the validator's checker
    check_batch.py              the validator's FULL format check (tokenizes like the trainer)
    train_step.py               ONE v8.3.3j step with the live trainer code -> candidate checkpoint
    gate_eval.py                the held-out gate (selection block, 3 head blocks, CRN candidate, score_ivw) on your GPUs
    simulate_turn.sh            all of the above, end to end
    prompts.py / grade.py / sampler.py / kitpath.py   shared helpers (pool, prompt renderings, grader, vLLM client, imports)
    tests/test_kit_cpu.py       CPU self-test of the whole chain (tiny model + fake vLLM; no GPU)
  validator/                    the live code (verbatim; MANIFEST.sha256)
    trainer-v833j/              ff_trainer.py, mimo_host_trainer_v83.py (sgd_step), v8211_batched.py (loss), v83_*.py, patched/
                                trainer-config.public.json = the live engine block + TrainerConfig + runtime versions
    engine/                     subnet/mimo_grpo (the engine the host trainer drives) + vendor/mimo_verl_pinned (GRPO advantage)
    batch/ff_format.py          the batch-file parser / checker / tokenizer the lifecycle uses
    gate/                       heldout.py (ivw-v2 scoring), gate_core.py (draw_sample, commitment), gate_service.py, gate-config.public.json
    probe/                      serve.sh (vLLM flags), probe_worker.py (sampling + grading), podctl.py, allan_src.py (lenient grader)
  data/original-math7496.tasks.json   the 7,496-task pool (sha256 77a4524abc279d0e6e95ec87d0e5604f501c8e409ac5ab656ecabf060ab50fe3)
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
5. **GREEN** means the candidate becomes the new head (bf16 + fp32 master). Everything about the turn is then published,
   starting from https://affine.io/k1/turns/<epoch>/green.json:
   - the baseline numbers (`/gate/heldout/<round>.json`: seed, selection block and head blocks per task, kept tasks, settings);
   - the probe results (`/gate/reveal/<epoch>.json`: head and candidate cells per task);
   - the tasks (`/k1/turns/<epoch>/tasks.json`: id, prompt, answer, level, kept);
   - the decision math (z, gain, SE, sigma_run, recomputed decision);
   - the winning file (`batch.json`) with its lr / kl (`turn.json`).

   **RED** means the head is kept and only the decision, z / gain, the counts and the file's sha256 are published.
   Your file, your lr/kl and your probe results stay private. While a round is open only its commitment is public.
   Only GREEN turns are revealed: a round that a GREEN closed publishes its seed, task ids, baseline and the GREEN
   candidate's per-task grades. RED turns and their probe results are never published, and a round that closes
   without a GREEN is never revealed. See https://affine.io/llms.txt section 7.
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
  (default 5e-7) and `kl` in [0.005, 0.1] (default 0.02). The bounds are inclusive and values are never clamped.
  A value out of range, a non-number or an extra key makes the file invalid (you're skipped, not charged).
  Putting `train` in a v1 file is also invalid.
- The file is UTF-8 JSON, at most 64 MiB. Prompts must be non-empty and distinct, and each needs at least one rollout.
- **prompt.** It's rendered as ONE user message: `apply_chat_template([{"role":"user","content":prompt}], add_generation_prompt=True)`
  with the head tokenizer. There's no system message, and the generation prompt is `<|im_start|>assistant\n`.
- **output.** It's encoded with `encode(output, add_special_tokens=False)`. `<|im_start|>`, `<|im_end|>` and
  `<|endoftext|>` (ids 248044–248046) are forbidden. Up to 8191 tokens, `<|im_end|>` is appended (it ended).
  Exactly 8192 tokens counts as capped (no EOS). More than 8192 is invalid. Prompt + output can be at most 12,288 tokens.
- **Caps.** At most 125 prompts and 1,000 rollouts (live value: mining.json `training.batch_cap`). Prompts past
  a cap are dropped whole, in file order.
- **reward.** Any finite number, used as given. Advantage = reward − the mean reward of that prompt's
  rollouts (no std normalisation), so **a prompt with one rollout gets advantage = its reward** (+1 for a correct
  reward-1 rollout). The trainer adds an overlong penalty of −0.5·clip((len−6000)/(8192−6000), 0, 1) to the reward.
- Check a file the way the validator does: `python3 kit/check_batch.py --hotkey <ss58> --tokenizer ./head batch.json`
  (`affine_batch.py check` does the structural part without a tokenizer).

## 3. The training step (v8.3.3j): what `kit/train_step.py` runs

This is one stateless step from the published head. All the code is in `validator/trainer-v833j` and `validator/engine`.
The live settings are in `trainer-config.public.json`.

- **Loss** (`v8211_batched.train_update_batched`): GRPO with one group per prompt, PPO clip 0.2, dual clip 3.0,
  and k3 (low-var) KL to the **reference = the initial checkpoint** `87de8304…` (the Hugging Face base
  `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`) with coefficient `kl`. It uses a derail mask (window 32, threshold −3, tail 64)
  and aggregates `seq-mean-token-mean` × 0.5635: L = 0.5635·(1/N)·Σ_i (1/K_i)·Σ_t [pg + kl·k3]. Gradients are
  accumulated in fp32 and globally clipped at 1.0. Every log-prob is recomputed from the head (temperature 1).
  The trainable tensors are `model.language_model.*` and `lm_head.*`.
- **Support renorm.** The PG term uses log q of your token renormalised over the head's own top-20 → top-p 0.95
  support. **A sampled token outside that support is dropped from the PG term only.** It still counts in KL and in
  the derail mask, and it's only logged (`support_renorm` in turn.json). Files are never rejected for it. The practical
  consequence: text the head would not sample (hand-written, or from another model) trains mostly nothing but KL.
  e368 had 0.14% of its tokens outside the support.
- **Step** (`mimo_host_trainer_v83.sgd_step`: fresh-adam, fp32 master, direction normgrad). For each tensor t, with
  W_t = the fp32 master:
  d_t = 5e-7·(g_t/(|g_t|+1e-8) + 0.01·W_t·[W_t ≥ 2-D and not a bias]);
  per-tensor cap c_t = 2e-5·(1−1e-6)·‖W_t‖; n0_t = min(‖d_t‖, c_t); with m = lr/5e-7, n_t = min(m·n0_t, c_t);
  the global cap is s = min(1, 1/‖n‖); Δ_t = s·n_t·g_t/‖g_t‖; M_C = M_H − Δ; and the candidate = bf16_rne(M_C).
  With the default lr, many tensors already sit at the per-tensor cap, so lr 1e-6 moves the model less than 2×
  (at e368, 242 of 427 tensors were at the cap; turn.json `training.train_params.step`).
- **Differences in the local run (execution only, same math):** it runs in one process with engine `padded`
  (the live run uses 8 GPUs, engine `packed`, fla 0.6.0 kernels), so it matches at the bf16-ulp level, not bit-exactly.
  Without `--master` the step bootstraps M_H = fp32(bf16 head). The live trainer uses the published fp32 master,
  whose manifest is at https://affine.io/k1/fp32-master/<head>.json.

```
python3 kit/fetch_head.py --out head && python3 kit/fetch_head.py --out ref --reference
python3 kit/train_step.py --head head --ref ref --batch batch.json --hotkey <ss58> --out cand --report step.json
```

## 4. The gate (held-out, ivw-v2): what `kit/gate_eval.py` runs

- **Pool:** the 7,496 tasks in `data/` (the list index is the task id). The gate prompt is
  `[system: MANDATE, user: problem]` with `enable_thinking=True`, where MANDATE =
  "Solve the math problem. Reason step by step in plain text, then end your response with the final answer in
  `\boxed{}`. Emit exactly one `\boxed{}` block and nothing after it."
- **Sampling:** temperature 1.0, top_p 0.95, top_k 20, 8,192 output tokens, 12,288 context, vLLM 0.32.0 on 8×B200
  (`probe/serve.sh`: `--dtype bfloat16 --max-model-len 12288 --enable-prefix-caching --max-num-seqs 512`). There's one
  request per sample, with seed = int(sha256(`P:affine_math:task:j`)[:12], 16).
- **Grader** (`kit/grade.py` = `probe_worker.grade`): a capped reply is neutral (unresolved). Otherwise the final channel
  after one `<think>…</think>` must hold exactly one balanced `\boxed{}`, which is checked with math-verify 0.9.0 against
  `\boxed{gold}`. If that check doesn't pass, the lenient fixes allan-lenient-v1 (P1/P2 projection, E1–E7
  equivalence; `probe/allan_src.py`) are tried.
- **A round.** The validator draws n = 1000 task ids with `gate_core.draw_sample(seed, 7496, 1000)` from a secret
  `os.urandom(32)` seed and publishes only a commitment (https://affine.io/gate/heldout/current.json). Block 0
  (the selection) runs the head k=4 on all 1000, and the **kept** tasks are those with 0 < c < 4. Blocks 1–3 are
  three more head runs on the kept tasks. Every candidate of the round runs k=4 on the kept tasks with **the same
  seeds as block 1**. The turn batch is not excluded. A round lasts until a GREEN, a head change or 8 decided turns.
  If a GREEN closes it, its seed, task ids, baseline and the GREEN turn's grades are revealed (`/gate/heldout/<round>.json`,
  `/gate/reveal/<epoch>.json`); RED turns are never revealed.
- **Score** (`heldout.score_ivw`): w_t = 1/(q(1−q)) with q = (c_sel+1)/(4+2); d_t = candidate rate − head rate over the 12
  head samples; gain = Σw·d/Σw; se_task = √(Σw²d²)/Σw; SE = √(se_task² + σ_run²·(1+1/3)), where σ_run = the run-to-run
  spread of the 3 head blocks (`heldout.run_sigma`; there's no floor since E354, and it was 0.022 for E271–E353); z = gain/SE.
- **GREEN iff** gain > 0 **and** z > 1.0 (strict) **and** the estimated whole-set cap-hit < 0.20 **and** there are ≥ 100 kept tasks.
  In practice that bar is about +1.5 to +2 points of weighted gain (GREEN e356 passed at gain 0.0198, z 1.027).

`gate_eval.py` draws its own local seed, so it can't know a live round's tasks. Your GPUs and vLLM build aren't the
gate's, so expect differences of a few points from run to run. Use `--baseline` to reuse a head baseline across candidates.

## 5. End to end

```
pip install -r miner/requirements-trainer.txt vllm==0.32.0 math-verify==0.9.0 httpx huggingface_hub
HOTKEY=5Your... bash miner/kit/simulate_turn.sh ./turn          # head -> rollouts -> batch -> step -> gate
python3 miner/affine_batch.py upload --hotkey 5Your... --key /private/miner.seed ./turn/batch.json
```

CPU self-test (no GPU, about 3 minutes):

```
pip install -r miner/requirements-sim-cpu.txt
python3 miner/kit/fetch_head.py --out /tmp/tok --tokenizer-only
python3 miner/kit/tests/test_kit_cpu.py --tokenizer /tmp/tok
```

## 6. How to win: what has worked so far (observations, not guarantees)

- **Every GREEN under the current rules** (e354 z 3.88, e356 z 1.03, e358 z 2.96, e360 z 2.42, e368 z 1.70, all on
  2026-10-31) was a **v1 file with the default lr and kl, 125 prompts × 1 correct rollout**. The prompts were MANDATE + "\n\n" +
  a pool problem (125/125 from the pool, mostly Level 4–5). Each rollout was the head's own sample with a
  `<think>` block and exactly one final `\boxed{}`. Since e342 almost every turn has had the same public counts
  (125 prompts × 1 correct rollout), including the REDs e362–e367 and e370–e376. Several of those REDs scored strongly
  negative (z −3 to −5.6 at e363, e365–e367, e370, e371). So the shape is common, but it isn't sufficient: which problems
  you pick and which samples you keep decide the sign.
- **Borderline difficulty.** The gate only scores tasks the head sometimes solves (0 < c < 4). Correct samples on
  problems the head usually fails or caps out on carry the signal. `make_batch.py --select correct1` picks those
  (hardest first, shortest correct sample).
- **Fewer capped replies.** In the performance eval (https://affine.io/performance.json, full pool) the pass rate went from
  0.8448 (E0) to 0.9014 (E368), while the cap-hit rate fell from 0.1175 to 0.0736. About 4.4 of those 5.7 points
  come from fewer replies hitting the 8,192-token cap, and the mean output length dropped from 2,043 to 1,769 tokens. Short correct
  rollouts (well under 6,000 tokens, where the overlong penalty starts) are consistent with that.
- **Sample from the head** with top_k 20 / top_p 0.95 / temperature 1. Off-support tokens are dropped from the PG term.
- The 1,000-rollout files with mixed outcomes (e339–e341, e343; z 0.70, −0.78, 0.44, −0.80) went RED, though those ran under the
  older sign step (before e354). No GREEN has used v2 `train` values yet (v2 is new on 2026-10-31).
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
| performance tile data | https://affine.io/performance.json (fixed full pool + private holdout; informational only) |
| step recipe / fp32 masters / replay | https://affine.io/k1/spec/step-recipe.json, /k1/fp32-master/<head>.json, /k1/replay/README.md |

## 8. Known gaps

- **fp32 master bytes.** The manifest (sha256 per file) is public, but the bytes aren't publicly downloadable yet.
  Without them `train_step.py` bootstraps from the bf16 head, which can differ slightly from the live step.
- **Trainer pod pip freeze.** Only the versions the trainer reports are published (torch, transformers, toploc, fla).
  `requirements-trainer.txt` pins those, plus what the code imports.
- **Packed 8-GPU kernels.** The local step runs `padded` in one process, which matches at the bf16-ulp level.
