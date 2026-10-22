# SN120 (Affine) v9 design exploration: pay for data by measured impact

**Author context:** for Jacob Steeves (owner). Written 2026-10-19 (America/Costa_Rica).  
**Status:** design exploration only. Nothing signed, activated, or put on chain.  
**Companion facts:** `/workspace/sn120-overfit-plan.md`, `/workspace/sn120-pay-rule-design.md`, `/workspace/allan-speedup/sn120-v824-train-config.md`, `/workspace/sn120-trainer-batching/REPORT.md`.

---

## 0. One-paragraph thesis

Miners today are paid for *rollouts of our policy on problems they choose*. That selects easy coverage (~15% of the 6,746 MATH pool), length shaping, and format games—not new capability. The proposal: **miners submit training data; the validator briefly fine-tunes a copy of the current checkpoint on each submission and pays by how much a private held-out score improves.** Copying is allowed if a challenger must beat the incumbent by an ε margin and eval variance is low. Self-reported quality is never trusted. The design below makes that economically and statistically feasible on our 9B MiMo + 8×H200 budget via a **tiered funnel** (cheap screens → LoRA probe → rare full confirmation).

---

## 1. Current state (v8.2.4) — lessons that bind v9

| Fact | Number / detail | Source |
|---|---|---|
| Environment | `affine_math`, Hendrycks MATH train, 7,496 tasks; mining pool **6,746**; reserve 750 | overfit-plan |
| What miners submit | Policy rollouts (group_size=8) on miner-chosen indices; earliest R2 win | pay-rule-design |
| Coverage | Cumulative distinct trained ≈ **14%**; e39–e43 submissions ≈ **15%** of pool | overfit-plan; user brief |
| Difficulty skew | Submissions ~**50% L1–2** vs ~**21%** of skipped; L5 ~**10%** vs ~**34%** skipped | user brief; overfit-plan trained 55% L1–2 / 8% L5 |
| Held-out | `fixed32`, 32 items, 1 sample; scores swing **~0.53–0.75**; SE ≈ **±0.09** | overfit-plan |
| What improved | Mostly **length / token-cap** behavior, not new reasoning | overfit-plan (wrong answers flat; unresolved grew) |
| Epoch wall time | ~**3 h** end-to-end; training job alone ~87 min (v8.2.3) → ~25–29 min projected (v8.2.4 batched) | REPORT / user |
| Trainer | **1×H200**, GRPO on verl; base **MiMo-family ~9B** (optimizer state ~107 GB measured) | REPORT.md “real 9B” |
| Train step shape | `max_tasks=128`, `group_size=8` → 1024 rollouts, ~1.8M tokens/epoch | REPORT |
| Owner compute | Targon org ~**$13.8k** credit; Lium; **8×H200** pods available for scoring | user |

**Implication for v9:** paying for *self-selected rollouts* will keep rewarding easy problems. Paying for *measured held-out Δ* forces miners to supply data that moves a private score—if and only if the score is low-noise and hard to game.

---

## 2. Prior art (only what is relevant)

### 2.1 Retrain / ablation valuation

- **Leave-one-out / Data Shapley** ([Ghorbani & Zou, ICML 2019](https://proceedings.mlr.press/v97/ghorbani19c.html)): equitable value = average marginal contribution over coalitions. Exact Shapley is exponential; MC needs many retrainings. **Relevant takeaway:** principled marginal value vs an *accepted pool* (diminishing returns, duplicates → near-zero value). Too expensive as a primary scorer for every miner submission; useful as the *definition* of value that cheaper proxies should approximate.
- **Direct ablation** (train with vs without the batch): gold standard for “did this data help?” Matches Jacob’s ask. Cost dominates unless the train is a short probe.

### 2.2 Gradient / influence proxies (cheap screens)

- **Influence functions** ([Koh & Liang, ICML 2017](https://proceedings.mlr.press/v70/koh17a.html)): upweighting effect via Hessian-inverse × grad. Hard at 9B (Hessian / IHVP). Use only as conceptual ancestor.
- **TracIn** ([Pruthi et al., NeurIPS 2020](https://arxiv.org/abs/2002.08484)): sum of LR-weighted grad dots across checkpoints; Hessian-free. Cheap relative to IF; noisy but good for ranking.
- **LESS** ([Xia et al., ICML 2024](https://arxiv.org/abs/2402.04333)): LoRA warm-up → low-dim Adam/LoRA gradient features → cosine match to *validation* gradients. Selecting 5% of data can beat full-set instruction tuning. **Directly relevant:** gradient alignment to held-out loss is a natural Stage-1 screen before paying for a real probe.
- **DsDm / datamodels** ([Engstrom et al., ICML 2024](https://proceedings.mlr.press/v235/engstrom24a.html); [Ilyas et al. 2022](https://arxiv.org/abs/2202.00622)): model how training-subset choice predicts target loss; ~2× compute multiplier vs random/quality filters on LM pretraining. **Relevant lesson:** human “quality” filters often lose to random; *model-aware* selection wins. Building full datamodels over miner streams is heavy; use the *framing* (optimize held-out utility) and TRAK-style proxies as screens, not as the payout oracle.

### 2.3 Learnability / curriculum (what data is worth training on)

- **RHO-LOSS** ([Mindermann et al., ICML 2022](https://proceedings.mlr.press/v162/mindermann22a.html)): prioritize points that are learnable, worth learning, and not yet learnt (reducible holdout loss). Avoids “hard = noisy” and “easy = already done.”
- **Online difficulty filtering for RLVR / GRPO** ([arXiv:2504.03380](https://arxiv.org/abs/2504.03380)): intermediate success rates (~0.2–0.8) maximize reward variance; balanced filtering up to **+12%** vs plain GRPO in fewer steps. Matches our v8 pain: miners farm p≈1 easy groups with almost no learning signal.
- **Sorscher-style pruning** ([NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/hash/7b75da9b61eda40fa35453ee5d657660-Abstract-Conference.html)): with scarce data keep easier; with abundant data keep harder. We are data-scarce on *useful* hard math → do not only reward hardest prompts.

### 2.4 Data competitions and data-market subnets

- **DataComp** ([Gadre et al., NeurIPS 2023](https://www.datacomp.ai/); [arxiv](https://arxiv.org/abs/2304.14108)): fix the *training code and compute*; compete on *data*. Score = held-out suite after standardized train. **Closest competition analog to SN120-v9.** Lesson: multi-scale compute tracks; forbid eval leakage; BYOD vs filtered-pool tracks.
- **Bittensor SN13 Data Universe** ([Macrocosmos docs](https://docs.macrocosmos.ai/subnets/subnet-13-data-universe/subnet-13-incentive-mechanism)): pays for *fresh, non-duplicate scraped bytes* × credibility—not model impact. Wrong objective for a frontier math model, but credibility/dedup patterns transfer.
- **Orion SN27** ([bittensor.ai/subnets/27](https://bittensor.ai/subnets/27); [SILX Orion](https://github.com/SILX-LABS/Orion)): campaign-driven data mining; validators replay hidden samples; score quality × useful tokens × availability. Still **judge/quality heuristics**, not held-out Δ on *our* trainer. v9 should differentiate by making **our private eval + our train recipe** the only oracle.

### 2.5 What we take into the mechanism

| Idea | Use in v9 |
|---|---|
| DataComp fixed-train / vary-data | Validator owns train+eval recipe; miners only supply data |
| Ablation / Shapley marginalism | Pay on Δ vs current accepted pool (not absolute score) |
| LESS / TracIn | Cheap Stage-1 gradient screen |
| RHO-LOSS + RL difficulty band | Prefer prompts with intermediate current pass rate |
| Orion credibility / SN13 dedup | Sybil + near-dup defenses, not the score itself |

---

## 3. Mechanism variants

### 3.1 What is submitted?

| Type | Pros | Cons | Verdict for v9 |
|---|---|---|---|
| **A. SFT traces** (prompt, CoT, answer) with verifiable answer | Simple LoRA/SFT probe; easy dedup | Can be teacher-distilled garbage that helps short SFT but not GRPO | **Primary format** |
| **B. RL prompts + verifier** (problem + checker only; no solution) | Matches our GRPO trainer; forces “edge of competence” | Need rollouts at probe time (costly); verifier gaming | **Secondary / preferred long-term** |
| **C. Preference pairs** | Good for length/style | Weak for math correctness; easy to fake | Defer |
| **D. Whole environments** (new task packs) | High upside; unconventional miner use Jacob wants | Huge qualify cost; contamination; one bad env poisons | **Invite-only / owner campaigns**, not open pay |

**Recommended submission schema (v9.0):**

```text
Submission = {
  format: "sft_math_v1" | "rl_prompt_v1",
  items: [{
    problem_id_or_text,
    solution_trace?,          # required for sft_math_v1
    final_answer,             # must pass math-verify
    meta: {source, license, difficulty_guess?}  # ignored for pay
  }],
  min_items: 8, max_items: 64,   # batch = unit of scoring
  content_hash, miner_hotkey, signature
}
```

Self-reported difficulty/meta is **ignored** for payout (assumption: miners fake anything self-reported).

### 3.2 How do we score?

#### (i) Per-submission ablation vs pooled leave-one-out

| | Per-submission ablation | Pooled LOO / Shapley-ish |
|---|---|---|
| Definition | Fine-tune incumbent on *only* this batch; measure Δ held-out | Value of batch *given* already-accepted pool |
| Cost | 1 probe / submission | Many probes, or sequential greedy |
| Gaming | Near-dup of a known good batch scores again | Near-dups get ~0 marginal value |
| Fit | Good for cold start | **Required once a pool exists** |

**v9:** Stage-2 probe scores **marginal Δ** against a frozen snapshot of the accepted pool (train on `pool ∪ batch` vs `pool`, or LoRA on batch after pool adapter). Approximate Shapley with **greedy online**: accept only if Δ > ε; pay proportional to clipped Δ.

#### (ii) Short LoRA probe vs full step

| | Short LoRA probe | Full GRPO / full FT step |
|---|---|---|
| Cost (9B, 1×H200) | **~0.05–0.25 GPU-h** / batch *(estimate)* | **~0.5–2+ GPU-h** *(estimate)* |
| Correlation with long training | Decent for local utility; can disagree | Gold for “hurts long training” |
| Use | Default pay signal | Rare confirmation / challenge |

**v9:** pay primarily on **LoRA probe Δ**; sample a fraction for full GRPO confirmation; slash or claw back if probe–full disagree badly (see attacks).

#### (iii) Gradient-alignment proxies (screen)

Stage-1: compute LoRA/last-layer grad of submission batch vs held-out loss grad (LESS-style). Keep top-K by cosine / TracIn score. Reject obvious negatives (anti-aligned). **No payout from Stage-1 alone**—only admission to Stage-2.

#### (iv) Marginal value, diminishing returns, dedup

- Exact / near-dup detection (MinHash + embedding cosine on problem text and on solution sketches).
- Pay `max(0, Δ − ε)` with ε set from eval SE (Section 5).
- Optional: exponential decay if problem embedding is within τ of an already-paid item.

---

## 4. Attack analysis and defenses

| Attack | Concrete pattern | Defense |
|---|---|---|
| **Eval contamination / leakage** | Infer held-out items from fine-grained public scores; overfit submissions to them | **Private rotating eval** (R300+ generated / DeepMath held-out); release only **coarse bins** (e.g. 5-level ordinal) or delayed exact scores; regenerate / rotate eval every N epochs; never mine on eval indices; canary items |
| **Copying / near-dup spam** | Resubmit others’ SFT with paraphrase | Allowed *as competition*: challenger must beat incumbent by **ε**; MinHash+embed dedup → marginal Δ≈0; content-addressed first-seen attribution (optional royalty split later) |
| **Sybil flooding** | Many hotkeys, tiny junk batches | Per-hotkey rate limit; Stage-1 reject before GPU; stake-weighted submission slots; bond burned on Stage-2 fail; credibility EMA (SN13-style) |
| **Probe-helps / long-hurts** | Data that overfits LoRA in 50 steps but poisons GRPO | Random **full-step audits**; pay escrow until audit; clawback if full Δ < 0; mix probe objectives (SFT + few GRPO steps) |
| **Poisoning / backdoors** | Triggers, wrong-but-verified answers, sleeper patterns | Native math-verify on all finals; canary eval; activation/grad anomaly screens; quarantine new acceptances for M epochs before main train mix weight ↑ |
| **Benchmark-specific overfitting** | Optimize only fixed32-like Level 4–5 style | Multi-suite private eval (in-domain R300 + MATH-500 + harder DeepMath slice); suite weights published; rotate |
| **Collusion** | Miners coordinate to share winning batch; validators leak eval | Single scoring authority (owner validator) for v9; encrypted eval blobs; multi-validator later with commit–reveal; no miner-side “quality judge” in the pay path (unlike Orion dual-judge) |

**Principle:** the only trusted signals are **(a)** verifier-checkable answers and **(b)** measured Δ on private eval under *our* train recipe. Everything else is a filter.

---

## 5. Noise and statistics

### 5.1 Why fixed32 fails as a pay oracle

Bernoulli SE ≈ √(p(1−p)/n). For p≈0.6, n=32, k=1: SE ≈ **0.09**. Observed swings 0.53–0.75 are noise-compatible. Cannot detect **+0.5 pp** (= 0.005).

### 5.2 Sizing for +0.5 pp

Treat per-item mean pass rate (k samples) as the unit; use **paired** evaluation (same draws / same problems) across base vs probe checkpoint.

| Design | n items | k samples | Approx SE on mean *(est.)* | Detectable Δ (rough, ~2 SE) |
|---|---|---|---|---|
| fixed32 today | 32 | 1 | ~0.09 | ~18 pp |
| R300, k=1 | 300 | 1 | ~0.028 | ~6 pp |
| R300, k=4 | 300 | 4 | ~0.015–0.025 | ~3–5 pp |
| **Pay eval target** | **800–1200** | **2–4** | **~0.01–0.015** | **~2–3 pp** |
| Research / offline | 2000+ | 4 | ~0.007 | ~1.5 pp |

**Honest limit:** detecting **+0.5 pp** per *single* submission at high confidence needs ~O(10⁴) effective independent trials—too expensive per batch. Therefore:

1. **ε margin for acceptance / copy-challenge:** set ε from measured paired SE, e.g. ε = **1.5–2 × SE_epoch** (estimate **ε ≈ 0.015–0.03** absolute with an 800×2 suite)—*not* 0.005.
2. **Pay on estimated Δ**, but require **Δ > ε** to earn nonzero weight (else copying and noise dominate).
3. **Sequential testing / aggregation:** accumulate a miner’s batches over a window; pay EMA of confirmed Δ; use sequential probability ratio tests to stop early on clear junk.
4. **Paired McNemar / cluster-bootstrap** across items for accept/reject; publish only coarse outcomes on-chain.

### 5.3 Seeds and variance

- Same forced draws for base and probe (as today’s `mimo-evaluation-common-random-draws-v1`).
- Optional: 2 seeds for Stage-3 confirmation only.
- Track unresolved rate separately so length games cannot masquerade as capability (lesson from v8).

---

## 6. Compute budget and tiered funnel

### 6.1 Measured anchors (9B on 1×H200)

From `/workspace/sn120-trainer-batching/REPORT.md`:

- Full epoch 128 tasks / 1024 rollouts / **1.79M tokens**: fwd/bwd **~14–42 min** depending on engine; full job historically ~87 min.
- Optimizer state **~107 GB**; model fits H200 141 GB with room for LoRA.

### 6.2 Cost estimates per scored submission *(all labeled estimates)*

Assume submission batch = 32 SFT items, mean 2k tokens/trace → ~64k tokens.

| Stage | Work | GPU-h / batch *(est.)* | Notes |
|---|---|---|---|
| **0. Admit** | schema, math-verify, MinHash dedup, rate limit | ~0 (CPU) | Reject most Sybil/junk |
| **1. Screen** | 1 forward + LoRA-grad alignment vs held-out grads (LESS-ish) | **0.01–0.03** | Batch many on 1 GPU |
| **2. LoRA probe** | r=16/32, 1–2 epochs, ~50–100 optimizer steps on batch; eval 800×2 | **0.08–0.25** | Dominant pay signal |
| **3. Full confirm** | short GRPO or full FT on batch∪pool slice + eval | **0.5–1.5** | 5–15% of Stage-2 passers |

**Blended cost per *submitted* batch** if Stage-0 kills 70%, Stage-1 keeps 20% of remainder, Stage-2 runs on those, Stage-3 on 10% of Stage-2:

- Per submission expected GPU-h ≈ `0.3×0.02 + 0.06×0.15 + 0.006×1.0` ≈ **0.02–0.04 GPU-h** *(estimate)*.
- Per **Stage-2 scored** (pay-eligible) batch: **~0.1–0.3 GPU-h** *(estimate)*.

### 6.3 Throughput on 8×H200

| Allocation | Role | Rough capacity *(est.)* |
|---|---|---|
| 2×H200 | Stage-1 screens | thousands of batches/day |
| 4×H200 | Stage-2 LoRA probes + eval | **~100–300 scored batches/day** at 0.15 GPU-h each |
| 1×H200 | Stage-3 confirms + canaries | ~10–30/day |
| 1×H200 | Main GRPO trainer (unchanged) | ~8–12 epochs/day if epochs ~2–3 h |

**Targon ~$13.8k:** at ~$3.5–5.8/H200-h, that is ~2.4k–4k GPU-h—enough for an offline pilot + weeks of Stage-2 if funnel is tight; **not** enough to full-ablate every miner upload without the funnel.

### 6.4 Funnel design (affordable)

```text
upload → verify+dedup → grad screen (top quantile) → LoRA probe Δ
       → if Δ>ε: escrow pay, add to quarantine pool
       → random full confirm → promote to main train mix or clawback
```

Daily caps: e.g. **150 Stage-2 slots/day** network-wide, stake-auctioned or first-credible-wins.

---

## 7. Recommended v9 design (concrete)

### 7.1 Submission format

- **Default:** `sft_math_v1` batches of **16–64** items; each final answer must pass **math-verify**; traces optional but boost probe (longer traces = more train tokens, still paid only on Δ).
- **Allowed later:** `rl_prompt_v1` (problem + verifier only); validator rolls out with current policy during probe (costlier—higher stake to enter).
- **Not paid:** preferences, raw scrapes without verifiers, self-reported scores.

### 7.2 Scoring pipeline

1. **Admit:** signature, size limits, verify answers, near-dup against paid corpus (threshold → reject or force marginal mode).
2. **Screen:** LESS-style gradient alignment to private eval loss; keep top fraction.
3. **Probe:** clone current main checkpoint; attach LoRA; train on batch (and optionally on small replay of accepted pool for marginalism); evaluate on **private Eval-V9**.
4. **Score:** `s = max(0, Δ_paired − ε)`; optional multiply by novelty ∈ (0,1].
5. **Confirm:** with probability p_c (e.g. 0.1), run short GRPO; if `Δ_full < 0` while `Δ_LoRA > ε`, mark unreliable, slash escrow.

### 7.3 Payout formula

Let `s_i` be the clipped marginal Δ for miner i’s accepted batches in the EMA window (e.g. 48–168 h, matching today’s hourly writer spirit).

```text
points_i = sum_batches max(0, Δ_b − ε) * novelty_b * credibility_i
weights  ∝ points_i          # no burn in v9.0; revisit
```

- **ε:** calibrated weekly from paired bootstrap SE of Eval-V9 (start **ε = 0.02** absolute).
- **credibility_i:** EMA of Stage-3 pass rate and audit validity (SN13-like exponent optional later).
- **Copying:** legal; near-dups get novelty→0 so s→0 unless they truly beat incumbent by ε on fresh eval.

### 7.4 Eval construction and rotation

| Suite | Size | Role |
|---|---|---|
| **Eval-V9-core** | **800** from MATH reserve ∪ stratified L1–5 (extend R300) | Paired pay metric, k=2 |
| **Eval-V9-hard** | **200** DeepMath / L5+ private | Capability; weight 0.3 |
| **Canaries** | 20–40 hidden | Leakage / poisoning detectors |
| **Public continuity** | fixed32 | Human dashboard only; **not** for pay |

- Rotate 10–20% of core every K epochs; hold out generation seed privately.
- Release to miners only **binned** Δ (e.g. {fail, small, medium, large}) or delayed exact Δ by 1–2 epochs.
- MATH-500 / public benches for *owner* dashboards, not miner optimization targets.

### 7.5 How accepted data feeds main training

1. Quarantine pool (weight 0) until Stage-3 or time M.
2. Main GRPO (or mixed SFT-pre then GRPO): sample tasks from  
   `α · miner_accepted ∪ (1−α) · owner_curriculum`  
   with α ramp 0→0.3 after pilot.
3. For `rl_prompt_v1`, prompts enter the existing group sampler with difficulty filter (pass rate band 0.2–0.8 under current policy)—directly attacking the easy-skew failure mode.
4. Keep length penalty / unresolved handling from v8.2.x so format games don’t dominate.

### 7.6 Rollout plan (offline pilot first)

| Phase | What | Compute *(est.)* | Gate |
|---|---|---|---|
| **P0** | Build Eval-V9 (800+200); measure SE; set ε | 1–2 H200-days | SE ≤ 0.015 on paired Δ |
| **P1** | Offline: score **our own** historical traces + synthetic hard sets + known bad dups with the funnel; correlate LoRA Δ vs full GRPO Δ | 3–5 H200-days | Spearman(LoRA, full) > 0.5; dups get Δ≈0 |
| **P2** | Shadow mode on mainnet: miners still paid v8.2.4; log v9 scores | 4×H200 ongoing | No critical infra bugs |
| **P3** | Split emissions (e.g. 20% v9 / 80% v8) | same | Held-out capability ↑ over 2 weeks |
| **P4** | Majority v9; deprecate rollout-pay except as bootstrap for new envs | — | Success metrics below |

### 7.7 Success metrics

1. **Capability:** Eval-V9-hard and MATH-500 pass@1 ↑ vs freeze baseline; not explained by shorter refusals alone (track unresolved separately).
2. **Coverage:** distinct trained problem embeddings / subjects ↑; L5 share of *accepted* data ≥ pool base rate.
3. **Integrity:** Stage-3 disagreement rate < 15%; canary catch rate on planted leaks; Sybil share of Stage-2 < 20%.
4. **Economics:** GPU-h per +1 pp confirmed < owner opportunity cost of hand-curating; blended ≤ **~0.25 GPU-h / scored batch**.
5. **Noise:** payment volatility from eval noise < payment from real Δ (check via scramble eval labels offline).

---

## 8. Top risks and open questions for Jacob

### Risks (ordered)

1. **LoRA–GRPO mismatch** — paid data helps the probe but not the 1×H200 GRPO run that actually ships. Mitigate with Stage-3 + clawback; if mismatch stays high, change the probe to “few GRPO steps” even if cost doubles.
2. **Eval leakage under emission pressure** — with ε small, miners will hunt the private set. Mitigate with rotation, coarse release, canaries; accept larger ε if needed.
3. **Thin honest supply** — producing verified hard SFT that moves 800-item eval is harder than farming easy rollouts; emissions may concentrate or idle. Mitigate with owner-seeded campaigns (Orion-style) and temporary hybrid pay.
4. **Cost blowup** — without strict Stage-0/1 caps, 8×H200 saturates on junk. Hard daily Stage-2 quota is mandatory.
5. **Political / miner break** — changing from rollout-pay to data-pay resets miner software. Shadow + split emissions required.

### Open questions

1. Should v9 **replace** rollout mining or run as **mech2** beside it (subnet dual-mechanism)?
2. Is **ε-copying with royalties** to first-seen author worth the accounting complexity?
3. Probe = **SFT-LoRA** or **mini-GRPO** on RL prompts from day one?
4. Do we open **environment submission** (type D) as bounties paid in TAO outside the automatic scorer?
5. Multi-validator scoring: when, and how to avoid validator collusion without re-introducing self-reported judges?
6. Exact MiMo checkpoint family variant and whether a **smaller proxy** (LESS transferable gradients) can screen for the 9B—pilot should measure transfer.

---

## 9. References (linked)

- Koh & Liang, Influence functions: https://proceedings.mlr.press/v70/koh17a.html  
- Pruthi et al., TracIn: https://arxiv.org/abs/2002.08484  
- Ghorbani & Zou, Data Shapley: https://proceedings.mlr.press/v97/ghorbani19c.html  
- Xia et al., LESS: https://arxiv.org/abs/2402.04333  
- Engstrom et al., DsDm: https://proceedings.mlr.press/v235/engstrom24a.html  
- Ilyas et al., Datamodels: https://arxiv.org/abs/2202.00622  
- Mindermann et al., RHO-LOSS: https://proceedings.mlr.press/v162/mindermann22a.html  
- Online difficulty filtering (RLVR): https://arxiv.org/abs/2504.03380  
- DataComp: https://www.datacomp.ai/ / https://arxiv.org/abs/2304.14108  
- SN13 Data Universe incentive: https://docs.macrocosmos.ai/subnets/subnet-13-data-universe/subnet-13-incentive-mechanism  
- Orion SN27: https://bittensor.ai/subnets/27  

---

## 10. Summary box

| Item | Recommendation |
|---|---|
| Pay for | Measured paired Δ on private Eval-V9 after LoRA probe, minus ε, × novelty × credibility |
| Do not pay for | Self-reported quality, easy rollout volume, unverifiable text |
| Model | MiMo-family **~9B**, 1×H200 GRPO main trainer |
| Cost / scored batch | **~0.1–0.3 GPU-h** Stage-2 *(est.)*; blended **~0.02–0.04** including rejects |
| Throughput | **~100–300** Stage-2 batches/day on 4×H200 *(est.)* |
| First step | Offline pilot on owned data + dups before any emission change |
