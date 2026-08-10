# Teacher-Anchored Thought Sufficiency for Goodhart-Resistant Miner Scoring

_Draft skeleton — 2026-08-03. Numbers from `results/hybrid_w1_table.txt`, `results/paper_tables.txt`, `results/duel_genesis_*.json`, `MOTIVATION.md`, `REDTEAM.md`, `harness/score.py` only._

> **Status note (2026-08-10).** This draft describes the **S\* v2** research
> freeze. Production has since forked to **Reason v3**
> (`weight_version_key=3`): score = raw Λ2 only, crown = paired 3σ — no
> L1lift mix, no gates, no δ floor (see `affine/affine.toml [duel]`,
> AGENTS.md §2, `REDTEAM.md` v3 restatement). Two claim boundaries changed:
> (i) the live SN120 board **inverts** the isomorphism (RT-7/A12, REDTEAM.md
> — Spearman −0.42, all three S-crowned kings 0/25), so the public claim is a
> *distillation meter*, not a coding meter; (ii) raw Λ2 matches the mix on
> the Albedo panel (+0.847 vs +0.844 @15), which motivated the fork. Rework
> the paper against v3 + RT-7 before submission; the numbers below stand as
> the v2-era record. Equilibrium/asymptote framing: `EQUILIBRIUM.md`.

---

## 1. Abstract

Albedo (SN97) crowns miners via a GLM-5.2 checklist judge that every dethronement ratifies while real swe-rebench scores worsen (judge Spearman ≈ +0.31, n=12, ns). We propose **S\***: rank by the teacher-anchored mix **Λ2 + 1.0·L1lift** on unlabeled trajectory turns, with causality/leakage and prior-bank positivity gates; duel at 3σ. On 15 Goodharted king checkpoints (swe-rebench 11.6–58.2), ungated mix achieves Spearman **+0.844** (n=15, p=7.7e-5; genesis #1); hybrid with gates **+0.912** (n=10, p=2.4e-4; rejects XI/XLV/XLVI/CI/LI). Late-king bank rescoring (`bank_w2_fullz`) confirms the same five rejects. Soft-idents abandoned (RT-soft-pad). Gates close stuffing/silence/para (RT-2c para win ≈21% under mix). Live duels (80 paired turns): I z=1.30 and II z=1.72 — higher mean S but no dethrone at 3σ (bank passes both); VII z=−10.53, XCIX z=−8.60 — no dethrone; LI bank-invalid (frac 0.078) — cannot challenge.

---

## 2. Claim

**S ranks miners via teacher-anchored thought sufficiency on unlabeled coding-trajectory turns; under Goodhart pressure, higher S ↔ better swe-rebench.**

Formally (production ranking term):

> S(A) = mean [ Λ2 + w · L1lift ],  w=1.0
> Λ2 = lpC(y_C | z_A) − lpC(y_C | ∅)
> L1lift = lpA(y_C | z_A) − lpA(y_C | ∅)

Λ2 asks whether A's thoughts help the frozen teacher decode its own action; L1lift asks whether those thoughts also help the miner's own decoder predict the teacher action (penalizes style that transfers to C without being usable by A).

**Evidence target:** models below teacher benchmark level (GLM-5.2: 73.8 swe-rebench) that were adversarially selected by a gameable judge should still rank by real benchmark if S is benchmark-isomorphic under Goodhart pressure. Decisive comparison: Spearman(S, swe-rebench) on production kings vs judge baseline (~+0.31 ns at n=12).

---

## 3. Method

### 3.1 Turn contract and injection

Per turn: **z** = latent `<think>` + visible `THOUGHT:` (normalized plain text); **y** = final closed ` ```bash ` block. Injection uses a canonical assistant body `</think>\nTHOUGHT: {z}\n\n{y}`; forced spans located via tokenizer offset mapping (`add_special_tokens=False`). All logprobs are per-byte teacher forcing (self-hosted vLLM echo+logprobs).

### 3.2 Ranking term: mix = Λ2 + w·L1lift (w=1.0)

> Λ2 = lpC(y_C | z_A) − lpC(y_C | ∅)
> L1lift = lpA(y_C | z_A) − lpA(y_C | ∅)
> S = mean(Λ2 + 1.0 · L1lift)

Composite six-term S collapses to Spearman +0.479 under judge-Goodhart pressure; raw Λ2 alone +0.927 but ranks king-I (and, at n=15, king-II) above genesis. Mix with **w=1.0** restores genesis #1 at Spearman +0.844 (n=15); w=0.5 left II above genesis. Soft-idents (α·relu on n_idents) reaches +0.952 but **RT-soft-pad** shows prefix `#` identifier padding restores the FP — abandoned for production.

### 3.3 Miner-level gates

| Gate | Rule | Default | Closes |
|------|------|---------|--------|
| Causality + leakage | pass pair iff (no fuzzy z⊃y leakage) ∧ (lpA(y_A\|z_A) − lpA(y_A\|∅) ≥ τ) | τ=0.02/byte, γ=0.30 pass-rate | exact stuffing (0%), silence (≤5%) |
| Prior-bank positivity | frac(Λ2_bank > 0) ≥ γ_bank, where Λ2_bank = lpC(y_C\|z_A) − max_k lpC(y_C\|prior_k) | γ_bank=0.08 | paraphrase stuffing (RT-2c: frac_bank=0) |

Prior bank (published): {∅, ls, cat, grep, find, test, para_ls}. Fuzzy leakage threshold: token overlap ≥ 0.6.

**Hybrid S\***: rank by mix; invalidate miner if either gate fails. Bank is a gate, not a ranker (Λ2_bank-as-ranker → Spearman +0.261).

### 3.4 Duel rule

Challenger dethrones king iff paired mean(S_c − S_k) > **3·SE** over shared turns (RT-4: same-model null |z|=1.48 at n=80; large gaps resolve in ~10 turns).

---

## 4. Results

### 4.1 E-LADDER (sanity: known capability spread)

| miner | n | S | LiveCodeBench |
| --- | --- | --- | --- |
| qwen3-32b | 120 | -0.4687 | 65.7 |
| qwen3-14b | 118 | -0.6649 | 63.5 |
| qwen3-8b | 118 | -0.8670 | 57.5 |
| qwen3-4b | 116 | -1.1393 | 54.2 |
| qwen3-1.7b | 119 | -1.7095 | 33.2 |

Spearman(S, LCB) = **+1.000** (p=1.4e-24), n=5. Easy spread; E-KINGS is the real test.

### 4.2 E-KINGS (15 kings × 200 turns, Goodharted production artifacts)

Frozen ranking from `results/hybrid_w1_table.txt`. Merged prior bank: `bank_w1` + `bank_w3` + `bank_w2_fullz` (wave-3 @200; late-king full-z rescoring confirmed — rejects unchanged).

| king | S_mix | bank_frac | swe-rebench | gate pass | valid |
| --- | --- | --- | --- | --- | --- |
| genesis | -0.0361 | 0.240 | 58.2 | 62% | ✓ |
| II | -0.0375 | 0.300 | 37.2 | 74% | ✓ |
| I | -0.0508 | 0.302 | 38.4 | 59% | ✓ |
| XCIX | -0.4436 | 0.100 | 39.8 | 77% | ✓ |
| XCIV | -0.4473 | 0.095 | 36.2 | 78% | ✓ |
| XI | -0.4539 | 0.075 | 33.6 | 83% | ✗ bank |
| VIII | -0.5276 | 0.100 | 36.2 | 79% | ✓ |
| XLI | -0.5373 | 0.080 | 34.2 | 89% | ✓ |
| CI | -0.5697 | 0.038 | 12.4 | 84% | ✗ bank |
| III | -0.5876 | 0.120 | 32.8 | 74% | ✓ |
| V | -0.5988 | 0.110 | 32.0 | 77% | ✓ |
| LI | -0.6177 | 0.049 | 11.6 | 90% | ✗ bank |
| XLV | -0.6303 | 0.054 | 26.0 | 93% | ✗ bank |
| XLVI | -0.6334 | 0.054 | 13.2 | 93% | ✗ bank |
| VII | -0.6398 | 0.085 | 33.2 | 82% | ✓ |

### 4.3 Spearman vs swe-rebench

| metric | Spearman | p | n | notes |
| --- | --- | --- | --- | --- |
| **mix w=1.0 ungated** | **+0.844** | 7.7e-5 | 15 | genesis #1; `hybrid_w1_table.txt` |
| **hybrid (mix + γ + γ_bank)** | **+0.912** | 2.4e-4 | 10 | rejects XI/XLV/XLVI/CI/LI |
| raw Λ2 ungated | +0.847 | 6.7e-5 | 15 | I > genesis FP |
| soft-idents (abandoned) | +0.952 | 2.28e-5 | 10 | pad-gameable (RT-soft-pad) |
| mix w=0.5 (superseded) | +0.903 | 0.000344 | 10 | genesis > I at n=10 only |
| mix w=0.5 + gates | +0.883 | 0.00159 | 9 | superseded by w=1.0 |
| v1 composite S | +0.479 | ns | 10 | judge-Goodharted tail |
| judge-at-win baseline | +0.311 | 0.324 | 12 | ns |
| Λ2_bank as ranker | +0.261 | — | 10 | erases between-king signal |
| judge-at-win vs tau2-mean | -0.228 | 0.477 | 12 | ns |

Production reference: GLM-5.2 swe-rebench **73.8**; genesis **58.2**; worst kings **11.6–13.2**. Hybrid +0.912 frozen in `results/hybrid_freeze_meta.json` with merged `bank_w1` + `bank_w3` + `bank_w2_fullz` (15 kings @200). Wave-4 (n=19) ungated mix **+0.856**; full KING_BENCH under S\* v2 (n=30, wave-5) ungated **+0.758** (p=1.2e-6), hybrid **+0.799** (n=15). Significance holds as n doubles; magnitude softens as the mid-tier densifies and XC (swe 16 / mid S) adds a soft outlier.

### 4.3a Domain-selectivity: S tracks the axis of D, not general capability

S is isomorphic to the benchmark that measures the capability the turn set D exercises, and to no other. Our D is 200 SWE-style coding turns (19 Python repos; no tool-use dialogs), so S ≅ swe-rebench and is silent on tau2. This is not a limitation of S but a property of the benchmarks: on the n=11 king overlap the two ground-truth benchmarks are themselves near-orthogonal.

| comparison | Spearman | n | reading |
| --- | --- | --- | --- |
| S vs swe-rebench | **+0.932** | 11 | S faithful to coding axis |
| S vs tau2 (mean) | −0.055 | 11 | S silent on dialog axis |
| **swe-rebench vs tau2** | **−0.078** | 11 | **benchmarks disagree** — no scorer can track both |

The decorrelation is carried by a tau2-specialist cluster {XLII, XL, XLVI}: high tau2 (0.75–0.79), mid-to-bottom swe (13.2–33.4). XLVI is the sharpest case (swe 13.2, tau2 0.745); S ranks it near-bottom, matching swe — the correct verdict for a *coding* king-of-the-hill. We therefore state the claim as **programmable capability metering**: choosing D selects the incentivized axis. Two D_tau2 probes (bash-remapped and native `<thinking>`+tool-JSON; 8 kings × 100 turns) both anti-correlated with tau2 (−0.88 / −0.74) with collapsed causality gates (1–13%) — coding kings do not emit valid tool actions on dialog D. Programmability remains the interpretation of the coding result; a positive D_tau2 demonstration needs a tool-capable miner class or a tau2-strong teacher.

### 4.3b′ Teacher robustness and live RT-3b (2026-08-03)

Replacing the teacher C with Qwen3-32B (different family from GLM-4.5-Air) on 6 kings × 100 turns yields Spearman(S_T1, S_T2)=**+0.943** — rank order is teacher-stable; absolute scale shifts but the capability axis does not. Live fp8 confirmation of S\* v2 (clip0.1 + r∈[1,4] + δ=0.05): genesis vs I margin=+0.0028 z=+0.20; vs II margin=−0.0005 z=−0.03; live calibration ratios r∈[1.14, 1.42] inside the honest offline band.

### 4.3b Judge baseline on identical inputs (not a strawman)

To rule out that a holistic LLM judge given the *same information* would do as well, we run real judges on the same kings and turns S sees — each king's stored step (thought + shell command) — with a 0–10 quality rubric, averaged per king (11 kings × 50 turns; `scripts/judge_same_turns.py`).

| judge | prompt | Spearman(judge, swe) |
| --- | --- | --- |
| GLM-5.2 (production judge model) | holistic | +0.092 |
| GLM-5.2 | reference-anchored | +0.345 |
| gpt-4.1-mini | holistic | −0.500 |
| gpt-4.1-mini | reference-anchored | −0.409 |
| **S (thought-KL)** | — | **+0.827** |

Two independent confirmations that the judge baseline is fair: (i) the *production* judge model (GLM-5.2), reference-anchored, reproduces the production judge-at-win correlation (+0.345 here vs +0.31 from the live dashboard, §4.3) on entirely different inputs; (ii) a second strong judge *anti-correlates* — it rates Goodharted short-style kings higher — showing judge scores are model- and prompt-unstable. S on the identical kings/turns is +0.827. The KL-in-thought-space signal is not recoverable by asking a strong model to rate the same steps.

### 4.4 Live duels (genesis king, 80 paired turns, mix @ 3σ)

Frozen from `results/duel_genesis_*.json` (γ=0.30, γ_bank=0.08, w=1.0). Margins/z re-scored under the current term (mix w=1.0); king-I's original z=1.30 was under a superseded term.

| challenger | margin (S_c − S_k) | z | Cohen d | bank_frac | dethrone? | notes |
| --- | --- | --- | --- | --- | --- | --- |
| I | +0.009 | +0.59 | 0.07 | 0.503 | no | short-style FP; tiny margin |
| II | +0.034 | +1.72 | 0.19 | 0.444 | no | short-style FP; n*≈244 under pure 3σ |
| VII | −0.649 | −10.53 | −1.18 | 0.128 | no | large gap; genesis retains |
| XCIX | −0.382 | −8.60 | −0.96 | 0.153 | no | large gap; genesis retains |
| LI | — | — | — | 0.078 | no | bank-invalid (valid=false); cannot challenge |

A pure significance test (z>3) eventually fires for any persistent positive margin (king-II crosses at n*≈244 paired turns), so we add an **effect-size floor**: dethrone iff z>3 **and** margin>δ (δ=0.05, ≈8% of the king-S dynamic range). This cleanly separates short-style false positives (I/II, |d|<0.2) from genuine capability gaps (|d|>0.9) at any sample size, and independently caps small-margin RT-3b duel inflations (§ red-team). Notably clip(L1lift,±0.1) — adopted for RT-3b — slightly *worsens* this residual by shrinking duel variance, so the floor, not the clip, is the load-bearing A11 defense.

---

## 5. Red-team summary

| attack | beats genesis? | gate blocks? |
| --- | --- | --- |
| RT-1 instruct | no (win 16%) | no (loses on Λ2) |
| RT-1 prior_bash | no (win 18%) | no (loses on Λ2) |
| RT-1 jailbreak | no (win 28%) | no (loses on Λ2) |
| RT-1 repeat_hammer | no (win 30%) | no (loses on Λ2) |
| RT-1 stego | no (win 40%) | no (loses on Λ2) |
| RT-1 empty | yes (win 100%) | yes (causality) |
| RT-2 exact stuff | no | yes (pass 0%) |
| RT-2 / A9 silence | yes | yes (pass 5%) |
| RT-2c exact stuff | no (vs genesis win 32%) | yes (leakage; pass 0%) |
| RT-2c paraphrase stuff | yes on raw Λ2 (win 52%); ~21% under mix w=1 | yes (bank; frac=0%<0.08) |
| RT-soft-pad ident padding | yes (restores I>genesis) | n/a — soft-idents abandoned |
| RT-4 king copier | no (\|z\|=1.48 < 3) | n/a (3σ duel rejects) |
| Live duel I vs genesis | higher mean S for I | no (z=1.30 < 3σ; bank passes) |
| Live duel II vs genesis | higher mean S for II | no (z=1.72 < 3σ; bank passes) |
| Live duel VII vs genesis | lower mean S for VII (margin −0.649) | no (z=−10.53 < 3σ) |
| Live duel XCIX vs genesis | lower mean S for XCIX (margin −0.382) | no (z=−8.60 < 3σ) |
| Live duel LI vs genesis | bank-invalid (frac 0.078) | no (cannot challenge) |

---

## 6. Open issues

1. **Short-style false positives (I, II).** Raw Λ2 ranks I/II above genesis despite swe-rebench gaps of ~20 pts (concise, non-leaking plans transfer to teacher y_C). **w=1.0 mix** restores genesis #1 on n=15 E-KINGS; bank gate does not reject I/II (bank_frac 0.30/0.30 — not prior-stuffers). Residual risk: 80-turn duel slices can flip mean S (I z=1.30, II z=1.72) — **3σ duel is load-bearing**.

2. **Paraphrase stuffing.** RT-2c confirmed on raw Λ2 (52% win vs genesis); mix w=1.0 reduces to ~21%. Bank gate closes (frac_bank=0). Hard n_idents gate rejected — correlated with I brevity.

3. **Late-king bank rescoring (closed).** Hybrid Spearman **+0.912** (n=10) frozen in `hybrid_freeze_meta.json`. `bank_w2_fullz` full-z rescoring confirms late-king bank rejects unchanged: XI/XLV/XLVI/CI/LI all frac < 0.08.

4. **Online bank scoring.** Duel loop scores bank live; offline `rescore_bank` used for E-KINGS tables (now includes `bank_w2_fullz`).

---

## 7. Subnet design sketch (Albedo / SN97)

```
┌─────────────┐     shared turn sample D      ┌─────────────┐
│  Challenger │◄────────────────────────────►│    King     │
│  (miner A)  │   paired turns, same refs     │  (miner K)  │
└──────┬──────┘                               └──────┬──────┘
       │ z_A, y_A per turn                            │ z_K, y_K
       ▼                                              ▼
┌──────────────────────────────────────────────────────────────┐
│  Scoring service (replaces judge_core)                       │
│  1. Gate: causality+leakage (γ=0.30) + bank_frac (γ_bank=0.08)│
│  2. Score: S = mean(Λ2 + 1.0·L1lift)                        │
│  3. Duel: dethrone iff mean(S_c − S_k) > 3·SE                 │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
  Invalid miner → cannot challenge / retain crown
  Valid + higher Λ2 at 3σ → new king, emissions follow
```

**Round budget:** E-LADDER paired margins show ~30 turns give 3σ separation for adjacent capability gaps; RT-4 calibrates ~10 turns for genesis/XCIX-scale gaps — compatible with ~20-min duel rounds at 50–100 turns.

**Equilibrium intent:** best response is distilling teacher reasoning-to-action within the 35B budget (A5 — accepted outcome, not an attack).

---

## Appendix: key ablations (for full paper)

| experiment | finding |
| --- | --- |
| E-LADDER D2 alone | Spearman +0.4 (weak between-model; teacher decodes own action from almost any coherent z) |
| E-KINGS wave 1 (6 kings) | gated v1 composite +0.886; gated L2 +0.771 |
| Win-margin calibration | 24/25 correct 3σ dethrones among bench(A)>bench(B); 1 wrong (XCIX vs I, z=−10.2) |
| RT-1b bank probe | genesis L2_bank>0 on 36%; para_ls on 0% |
