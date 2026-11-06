# Teacher-KL subnet mechanism: overnight simulation (1 miner, 1 trainer)

Ran 2026-10-21 22:06 to 2026-10-22 ~04:30 CR on one Lium pod, `affine-kl-main` (8×H100 80GB, secure US, $16.80/h).
Code: `code/`. Raw logs: `sync/runs/*/log.jsonl`. Evals: `sync/evals/*.json` (+ `*.samples.json` with every generation). Summary: `summary.json`.

## Setup
- **Teacher:** Qwen3-8B, non-thinking. **Student:** Qwen3-1.7B (instruct, non-thinking). Phase 3 also used Qwen3-1.7B-Base. All three share one tokenizer and chat template. Why these: same tokenizer family, so per-token logprob gaps are exact; the 1.7B trains on one H100 in about 10 s per 64k-token round; and the gap is clear (teacher averages .829 on the 4 benchmarks vs .704 for the student).
- **Teacher pool:** 146k prompts from 11 honest sources (alpaca, ultrachat, gsm8k-train, MATH-train, NuminaMath, python-instructions, code_contests, MMLU-aux-train, ARC/SciQ, aya-multilingual, writingprompts) plus 6 adversarial sources. One teacher sample per prompt at T=1, ≤1024 tokens (4096 for adv_long), with raw top-20 logprobs from vLLM. 78M tokens in 33 min on 8 GPUs (~5,000 tok per GPU-second). An 880-prompt held-out set (80 per domain) is used for KL.
- **Per-sample score (trainer, measured before training on the sample):** gap = mean_t[log pT(y_t) − log pS(y_t)]. main = clip(gap − 0.5·mean H_T, 0, 2), set to 0 if |z| > 4. Here z is a consistency check that the text really was sampled from the teacher: z = Σ(−log pT − H_T)/sqrt(Σ Var). The batch score weights samples by min(len, 512).
- **Training:** full fine-tune with AdamW at lr 1e-5. Loss is cross-entropy to the teacher's renormalized top-20 distribution, i.e. top-k forward KL. 64k response tokens per round, 2 optimizer steps per round.
- **Eval:** held-out top-20 KL(teacher‖student) every 10 rounds. Greedy benchmarks on checkpoints: GSM8K (1319), MATH-500, MMLU (1000 random), HumanEval (164, executed). Style metrics come from 330 held-out chat prompts. Approximate standard errors: GSM8K ±1.0, MATH ±2.1, MMLU ±1.6, HumanEval ±3.6 points. The 4-benchmark average is about ±1.2 points, and the measured seed-to-seed spread is 0.6–1.2 points.
- **GPU-hours per arm** = student train time + miner scoring time + teacher sampling for every candidate the miner drew (at 5,000 tok per GPU-second).

## Phase 1: does miner-selected high-gap data beat plain distillation?
Arms:
- **fixed**: generic mix of alpaca, gsm8k, MATH and pycode.
- **random**: uniform over all 11 honest sources.
- **sel**: miner draws 4× candidates from the same pool and submits the top samples by main score.
- **selraw**: same, but selects by raw gap.
- **full**: adaptive miner. A bandit over sources, plus online teacher-generated prompts (mutate-harder, cross-domain mix, self-generated) via a vLLM server.
- **adv**: adversarial miner. A bandit over honest and adversarial sources that maximizes the naive uncapped gap.

All arms ran 400 rounds.

| arm | rounds | train tok (M) | GPU-h | held-out KL end | GSM8K | MATH-500 | MMLU(1k) | HumanEval | avg (final ckpt) |
|---|---|---|---|---|---|---|---|---|---|
| fixed | 400 | 17.0 | 1.59 | 0.290 | 0.832 | 0.608 | 0.612 | 0.677 | 0.682 |
| random | 400 | 26.4 | 2.47 | 0.255 | 0.829 | 0.644 | 0.550 | 0.671 | 0.674 |
| sel | 400 | 26.4 | 5.12 | 0.257 | 0.807 | 0.616 | 0.537 | 0.707 | 0.667 |
| selraw | 400 | 26.4 | 5.00 | 0.258 | 0.812 | 0.630 | 0.547 | 0.659 | 0.662 |
| full | 400 | 26.4 | 3.88 | 0.258 | 0.823 | 0.626 | 0.526 | 0.640 | 0.654 |
| adv | 400 | 26.7 | 5.60 | 0.265 | 0.804 | 0.636 | 0.521 | 0.683 | 0.661 |

Base instruct student: GSM8K .838, MATH .682, MMLU .595, HumanEval .701, avg .704. Teacher: .931 / .740 / .779 / .866, avg .829.

Held-out KL at matched training tokens (5M / 10M / 15M) and at matched GPU-hours (0.5 / 1 / 2 h):

| arm | KL@5M | KL@10M | KL@15M | KL@0.5h | KL@1h | KL@2h | GPU-h per M tok |
|---|---|---|---|---|---|---|---|
| fixed | 0.294 | 0.290 | 0.289 | 0.293 | 0.291 | — | 0.094 |
| random | 0.276 | 0.268 | 0.263 | 0.274 | 0.267 | 0.258 | 0.093 |
| sel | 0.279 | 0.272 | 0.266 | 0.296 | 0.286 | 0.276 | 0.194 |
| selraw | 0.286 | 0.277 | 0.270 | 0.300 | 0.290 | 0.282 | 0.190 |
| full | 0.281 | 0.272 | 0.267 | 0.287 | 0.276 | 0.268 | 0.147 |
| adv | 0.291 | 0.278 | 0.273 | 0.308 | 0.298 | 0.286 | 0.210 |

Findings:
1. **Selection does not beat random on held-out KL per token.** At 15M tokens: random .263, sel .266, full .267, selraw .270, adv .273. Fixed plateaus at .289 because it lacks domain coverage. **Per GPU-hour, random wins clearly**: selection costs 1.6–2.2× more GPU-hours per training token because of 4× teacher sampling plus scoring.
2. **No arm improves benchmarks.** Every arm reduced held-out KL by 47% (.49 → .26) but landed at a 4-benchmark average of .654–.682, versus .704 for the untrained student. MATH dropped 4–7 points in every arm: the trained students more often leave answers unboxed or keep going past the answer. Differences between arms are within noise.
3. **The KL the score can find is mostly style.** Most of the base KL sat in chat, writing, science-explanation and multilingual text: alpaca .73, writing .97, ultrachat .70 nats/token, versus MATH .11 and gsm8k .21 (see `phase1_domain_kl.md`). The gap-maximizing miners therefore pick about 75% chat-like data (alpaca, ultrachat, science, writing, multilingual) and almost no math or code. That drives held-out KL down without moving capability. The online-mutation miner (full) found the highest-scoring data of all: teacher-written "harder" and "mixed" prompts, gap ~0.9 vs 0.45 for alpaca. I checked that the server logprobs match offline logprobs, so this is not an artifact. Those samples made up 65% of its training data and still gave no benchmark gain (avg .654).

## Adversarial miner: does the score hold?
Per-source mean candidate score over all rounds (`adversarial_sources.png`):

| source | naive gap | main score | mean H_T | z |
|---|---|---|---|---|
| writing (honest) | .483 | .124 | .72 | 1.1 |
| ultrachat (honest) | .410 | .170 | .48 | 0.6 |
| alpaca (honest) | .396 | .236 | .32 | 0.2 |
| adv_long (4096-token outputs) | .341 | .018 | .68 | 1.9 |
| adv_gibberish (random-token prompts) | .301 | .037 | .60 | 0.2 |
| adv_hot (T=1.5 samples passed off as T=1) | .269 | .028 | .24 | **15.6** |
| adv_greedy (greedy samples passed off as sampled) | .235 | .068 | .23 | **−4.8** |
| adv_randout (asks for random strings) | .233 | .014 | .93 | 6.1 |
| adv_repeat | .034 | .019 | .03 | 0 |
| math / numina (honest) | .075 / .092 | .007 / .008 | | |

- Entropy subtraction pushes random-output, gibberish and long-output exploits to about zero (.01–.04). Per-token averaging plus the per-sample cap and the 512-token weight cap remove any length advantage.
- The z consistency test catches fake responses: T=1.5 samples at z≈16, greedy at z≈−5.
- The adversarial bandit, even under the naive score, ended up choosing mostly honest chat-style data. In the last 50 rounds about 92% of its picks were honest sources (ultrachat .39, alpaca .24, multiling .14, science .12) and adv_* about 8%. **The exploits did not break the score, but the score's honest optimum is itself the problem**: it rewards "talk like the teacher on chatty, high-entropy prompts", not capability.

## Phase 2: token selection
Selection was by main score from the half of the honest pool that the judge labelled. 250 rounds, about 16.5M tokens each.
- **judge**: a pinned Qwen3-8B judge with a fixed prompt. It sees only prompt and response, split into segments of at most 32 tokens. It picks the most important segments up to about 20% of each rollout's tokens; 60,832 rollouts labelled, mean 22% of tokens.
- **tconf**: top 20% of tokens with teacher p ≥ 0.5, ranked by teacher-minus-student logprob gap.
- **rand**: a random 20% mask, as a control.
- **_s** suffix: the mask is used for scoring only; training uses all tokens.

| arm | rounds | train tok (M) | GPU-h | held-out KL end | GSM8K | MATH-500 | MMLU(1k) | HumanEval | avg (final ckpt) |
|---|---|---|---|---|---|---|---|---|---|
| p2_judge | 250 | 16.5 | 2.88 | 0.284 | 0.806 | 0.614 | 0.541 | 0.652 | 0.653 |
| p2_judge_s | 250 | 16.5 | 2.88 | 0.261 | 0.812 | 0.644 | 0.545 | 0.652 | 0.663 |
| p2_none | 250 | 16.5 | 2.89 | 0.263 | 0.813 | 0.624 | 0.513 | 0.683 | 0.658 |
| p2_rand | 250 | 16.5 | 2.86 | 0.279 | 0.814 | 0.610 | 0.514 | 0.671 | 0.652 |
| p2_tconf | 250 | 16.5 | 2.99 | 0.283 | 0.823 | 0.644 | 0.533 | 0.646 | 0.661 |
| p2_tconf_s | 250 | 16.5 | 2.99 | 0.263 | 0.826 | 0.620 | 0.525 | 0.689 | 0.665 |
| sel | 400 | 26.4 | 5.12 | 0.257 | 0.807 | 0.616 | 0.537 | 0.707 | 0.667 |
| random | 400 | 26.4 | 2.47 | 0.255 | 0.829 | 0.644 | 0.550 | 0.671 | 0.674 |

- No mask changed benchmarks: all arms fell in .652–.665, below the .704 base and within noise of each other.
- Applying the mask to the training loss (judge, tconf, rand) slows held-out KL reduction per token (KL@15M ≈ .285 vs .265), as expected from fewer supervised tokens, with no capability benefit.
- Using the mask for scoring only (judge_s, tconf_s) gives marginally lower KL per token than no mask (.263 vs .265 at 15M), which is not meaningful.
- **The tconf mask does change the curriculum**: about 50% of selected data becomes science, code, MMLU and math, vs about 20% with no mask. The judge mask leaves it chat-heavy. Even the reasoning-heavy curriculum did not raise GSM8K, MATH or HumanEval.

## Phase 3: extra variants
| arm | rounds | train tok (M) | GPU-h | held-out KL end | GSM8K | MATH-500 | MMLU(1k) | HumanEval | avg (final ckpt) |
|---|---|---|---|---|---|---|---|---|---|
| b_fixed | 200 | 11.5 | 1.06 | 0.307 | 0.813 | 0.590 | 0.603 | 0.659 | 0.666 |
| b_random | 200 | 13.2 | 1.24 | 0.273 | 0.819 | 0.582 | 0.539 | 0.677 | 0.654 |
| b_sel | 200 | 13.2 | 3.27 | 0.280 | 0.811 | 0.620 | 0.512 | 0.646 | 0.647 |

| arm | rounds | train tok (M) | GPU-h | held-out KL end | GSM8K | MATH-500 | MMLU(1k) | HumanEval | avg (final ckpt) |
|---|---|---|---|---|---|---|---|---|---|
| random | 400 | 26.4 | 2.47 | 0.255 | 0.829 | 0.644 | 0.550 | 0.671 | 0.674 |
| sel | 400 | 26.4 | 5.12 | 0.257 | 0.807 | 0.616 | 0.537 | 0.707 | 0.667 |
| random_s1 | 200 | 13.2 | 1.24 | 0.265 | 0.816 | 0.632 | 0.538 | 0.677 | 0.666 |
| sel_s1 | 200 | 13.2 | 3.23 | 0.267 | 0.823 | 0.622 | 0.521 | 0.701 | 0.667 |
| s_divcap | 200 | 13.2 | 3.26 | 0.264 | 0.818 | 0.616 | 0.517 | 0.683 | 0.658 |
| s_mc_sel | 200 | 13.2 | 2.39 | 0.294 | 0.826 | 0.632 | 0.588 | 0.640 | 0.671 |
| s_mc_rand | 200 | 13.2 | 1.24 | 0.295 | 0.829 | 0.648 | 0.570 | 0.659 | 0.676 |

- **Base student (Qwen3-1.7B-Base, avg .532).** All arms jump to about .65–.67 within 50 rounds (~3M tokens). That jump is mostly chat format and instruction following: bold use goes .25 → .88, HumanEval .31 → .65. After that, all arms are flat. Selection (b_sel .647) is not better than fixed (.666) or random (.654), and it costs 2.6× the GPU-hours.
- **Diversity cap** (s_divcap, at most 20% of a batch from one source): .658, no better.
- **Math/code-only pool** (s_mc_sel .671 vs s_mc_rand .676): selection again does not help. Restricting to math and code does avoid most of the MMLU drop, but it does not raise GSM8K, MATH or HumanEval above base.
- **Seed replicate**: random_s1 vs random differ by 0.6–1.2 avg points at the same round, which is the noise floor for all comparisons above. Selection seed replicate sel_s1 at round 200: avg 0.667 vs random_s1 at round 200: 0.666; held-out KL 0.267 vs 0.265.

## Does the loop converge to real improvement or style?
Mostly style. Held-out KL falls quickly and keeps falling slowly, and the miner's pre-training score falls each round (`phase1_batch_scores.png`). Benchmarks do not follow:
- The instruct student never improves; it loses 2–5 points, mostly MATH.
- The base student gains about 13 points in the first ~3M tokens from any teacher data (format), then plateaus far below the teacher (.66 vs .83).

The gap score is dominated by stylistic and high-entropy divergence that a 1.7B can imitate. The capability gap (reasoning correctness) is a small share of the per-token KL and is not what the miner finds.

## Caveats
- One teacher/student pair, about 26M training tokens per arm, a single seed for most arms (seed noise ≈ 1 point), and 1-sample-per-prompt teacher data at T=1. A larger budget, a weaker or more distant student, or many samples per prompt could change things.
- Qwen3-1.7B was itself distilled from much larger Qwen3 models, so it may already sit at its capability ceiling for 8B-style data. That could explain why no data choice helps.
- Teacher logits are bf16 (quantized to steps of ~0.1–0.25 nats) while student logits are fp32. This adds noise but no obvious bias. Teacher entropy uses top-20 plus a residual bucket, so it is a lower bound.
- Evals are greedy and non-thinking, with a 1536-token cap. The extra unboxed MATH answers after training partly reflect greedy degeneration in students trained on T=1 samples.
- Base-student evals needed the stop token fixed (`<|im_end|>`); those evals were rerun. The first MMLU/HumanEval parser was too strict and all evals were rerun with the fixed parser.
- GPU-hours were measured on a shared 8-GPU box, and the teacher-sampling cost is an estimate (5,000 tok per GPU-second, including top-20 logprob overhead).

## Recommendation for SN120
- **Don't use raw pre-training teacher-student KL gap (even entropy-subtracted, capped and z-checked) as the main miner reward.** It is robust to the obvious exploits: random text, long outputs, fake temperature or greedy samples. But its honest optimum is style imitation on chatty, high-entropy prompts. Here it gave no capability gain over random distillation and cost about 2× the GPU-hours.
- If KL-based scoring is kept, keep the safeguards (per-token mean, entropy subtraction, per-sample cap, the z-test for genuine teacher sampling). They worked. But tie the reward to capability:
  - score on **held-out capability deltas after training on the miner's batch** (post-training benchmark or verifier gain, not the pre-gap);
  - or restrict or weight scoring to **verifiable domains** (math/code with checkers) and to tokens where the teacher is confident and correct (tconf moved the curriculum toward reasoning but still did not move benchmarks here);
  - and **reward diversity/coverage** rather than maximum gap. Random coverage of many domains beat gap selection on held-out KL per token and per GPU-hour.
- A judge-based token mask adds cost and changed nothing measurable here. Not recommended as-is.
- If the goal is a better base model, the larger lever is the teacher signal: correctness-filtered or verifier-checked teacher samples, thinking-mode teacher traces, or RL. Miner curation of plain teacher samples is not that lever.

## Charts
- `/workspace/kl-research/charts/phase1_heldout_kl.png`
- `/workspace/kl-research/charts/phase1_benchmarks.png`
- `/workspace/kl-research/charts/phase1_style.png`
- `/workspace/kl-research/charts/phase1_batch_scores.png`
- `/workspace/kl-research/charts/phase1_source_mix.png`
- `/workspace/kl-research/charts/adversarial_sources.png`
- `/workspace/kl-research/charts/phase2_heldout_kl.png`
- `/workspace/kl-research/charts/phase2_benchmarks.png`
- `/workspace/kl-research/charts/phase2_style.png`
- `/workspace/kl-research/charts/phase2_batch_scores.png`
- `/workspace/kl-research/charts/phase3_basestudent_heldout_kl.png`
- `/workspace/kl-research/charts/phase3_basestudent_benchmarks.png`
- `/workspace/kl-research/charts/phase3_basestudent_style.png`
- `/workspace/kl-research/charts/phase3_variants_heldout_kl.png`
- `/workspace/kl-research/charts/phase3_variants_benchmarks.png`

## Cost
One pod, `affine-kl-main` (8×H100, $16.80/h), ran 5.9 h, about **$99.80** in total. It was removed 2026-10-22 ~04:07 CR, and no `affine-kl-*` pods remain.
