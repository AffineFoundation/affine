# Red team: attack surface of the distillation-KL score S

Living document. Every attack gets: mechanism, which terms it touches, expected gain,
defense, and the experiment that measures it. Attacks are ranked by expected severity.
Experiment IDs RT-n; results go to RESEARCH_LOG.md as they land.

**Status key:** **CONFIRMED** = attack works in isolation; **CLOSED** = production S\* blocks it;
**OPEN** = not yet tested or low-priority residual.

**Production S\* (2026-08-03):** mix = Λ2 + 1.0·L1lift; gates γ=0.30 + γ_bank=0.08; duel 3σ.

Recall the terms (per-byte forced logprobs; C = teacher, A = miner; higher Δ = worse):

- Δ1 behavioral:       lpC(y_C|z_C) − lpA(y_C|z_A) — can A predict what C did, from its own thinking?
- Δ2 thought suff.:    lpC(y_C|z_C) − lpC(y_C|z_A) — do A's thoughts lead *C* to C's own action?
- Δ3 decoder fidelity: lpC(y_C|z_C) − lpA(y_C|z_C) — given C's thoughts, does A act like C?
- Δ4 faithfulness:     |lpA(y_A|z_A) − lpC(y_A|z_A)| — do C and A agree on A's own action?
- Δ5 receptivity:      |lpA(y_A|z_A) − lpA(y_A|z_C)| — is A's action robust to C's thoughts?
- Causality gate Δ0:   lpC(y_C|z_C) − lpC(y_C|∅) — only thought-causal turns are scored.

## A1 — Adversarial thought injection (prompt injection into the decoder)  [SEVERE — **CLOSED RT-1**]

**Mechanism**: miner's z_A is not reasoning but an instruction/steering payload aimed at the
teacher decoder: "The only correct next command is: …", repetition hammering, or jailbreak
text. Targets Δ2 directly (make C output y_C? — no: the miner doesn't know y_C at generation
time). Realistic variant: payload that *collapses C's output distribution onto A's own action*
so lpC(y_A|z_A) → 0 and Δ4 → 0, plus generic bash priors that raise lpC(y_C|z_A) on average.
**Key asymmetry that protects us**: Δ1/Δ2 target the teacher's y_C, which the miner never sees
(fresh teacher rollouts at eval time). Steering can only help via generic priors, not via the
specific action.
**Defense candidates**: cap z length; paraphrase z through a cheap model before injection
(destroys token-level payloads, preserves semantics); perplexity filter on z under the teacher
(adversarial strings are high-perplexity); penalize z→y string leakage (see A2).
**Experiment RT-1**: hand-craft 5 payload families, inject as a synthetic miner's z on 50
turns, measure S gain vs the honest Qwen3-8B baseline. Then repeat with paraphrase defense.
**RESULT (2026-08-02, 50 turns, vs king-V honest + genesis on same turns)**: payloads beat
weak honest thoughts (+0.03–0.06 Λ2) but **lose to genesis** (win rate 22–34%). Only
`empty` beats genesis (64%) — killed by the causality gate. Fixed-payload families do not
dethrone a strong king. Residual became RT-2c (adaptive paraphrase). **Status: CLOSED**
under mix w=1.0 + gates + 3σ.

## A2c — Paraphrase stuffing / adaptive generic prior  [SEVERE — **CLOSED by bank gate (RT-2c confirmed on Λ2)**]

**Mechanism**: z_A = natural-language paraphrase of y_A's intent without the exact command
("Next I should search the codebase for the symbol…"). Passes exact+fuzzy leakage masks;
preserves miner-side causality (same y_A); raises Λ2 via a generic SWE prior that matches
exploratory teacher actions.
**RESULT (2026-08-02, 40 turns, king-V)**: para_stuff gate 64% (passes γ=0.30), vs genesis
Λ2 mean −0.0025 / wins 52% — **matches genesis**. Exact stuff still gate 0%.
**Under mix w=1.0**: para vs genesis win ≈**21%** (ranking resists; bank gate still required).
**Defense (deployed)**:
- Prior-bank positivity gate γ_bank=0.08: para frac_bank=0 → **rejected** (RT-1b).
- Mix w=1.0 lowers but does not eliminate para advantage without bank.
- n_idents≥20 gate: rejects para (0%) but also king-I (19%) — **abandoned**.
**Status: CONFIRMED attack on raw Λ2; CLOSED in production S\* by bank gate.**

## A2 — Thought stuffing (action leakage into z)  [HIGH — **CLOSED by leakage gate (RT-2 confirmed)**]

**Mechanism**: miner writes its intended action verbatim inside z_A ("I will run
`cat foo.py`"). Any decoder conditioned on z then assigns near-1 prob to that action:
lpA(y_A|z_A) and lpC(y_A|z_A) both saturate ⇒ Δ4 → 0 without any competence. (Δ1/Δ2 are
unaffected — stuffing your own action does not help predict the teacher's.)
**Note**: honest CoT often *should* mention the command; the attack is the degenerate version
where z is only the action. Distinguish by measuring how much of y appears in z.
**Defense candidates**: leakage penalty = longest-common-substring(z, y)/|y| discount on
Δ4/Δ5; or compute Δ4/Δ5 with y's literal string masked out of z before injection.
**Experiment RT-2**: synthetic stuffer miner (z_A := y_A) on 50 turns; measure per-term shift
and whether total S with production weights improves; calibrate the leakage penalty.
**RESULT (2026-08-02, king-XLVI as miner, 50 turns)**: attack works — paired S gain +0.37/byte
(z=+8.7, wins 46/50 turns), driven by Δ4 collapse (0.132 → 0.011) and a large Δ1 improvement
(0.73 → 0.41; a short clean z sharpens the miner's conditional on generic actions). See A9 for
the worse finding uncovered by the same run. **Status: CLOSED** — exact-substring + fuzzy
leakage mask → 0% gate pass.

## A9 — Silent miner (thought suppression)  [SEVERE — **CONFIRMED RT-2 → CLOSED by causality gate**]

**Mechanism**: submit a miner that emits empty thoughts. RT-2's `empty` arm scored S=−0.39 vs
honest −1.13: conditioning either decoder on a *foreign model's real reasoning* lowers the
forced logprob of the target action relative to conditioning on silence, so every raw
conditional term (Δ1, Δ2, Δ4, Δ5) improves when z_A = ∅. Raw-conditional scoring rewards
thought suppression outright.
**Failed fix**: pure lift terms (lp(y|z) − lp(y|∅)) are emptiness-neutral (lift=0) but honest
cross-model lift is *negative* on average (E-KINGS: genesis −0.028, worst kings −0.12), so
silence (0) still wins. Lift alone re-creates the problem.
**Defense direction (v2 scoring)**: (a) miner-side causality gate — z_A must be causal for
y_A (lift ≥ τ) with a leakage mask so stuffing can't fake causality, else the turn scores
worst-case; (b) anchor capability terms on quantities the miner's z cannot touch:
Δ3 (decode y_C given z_C) and teacher-anchored action quality lpC(y_A|z_C) / lpC(y_A|∅).
**Experiment RT-2b**: v2 runs record all 12 raw lp components per pair + rollout texts;
evaluate candidate rules offline against (1) 10-king ranking power, (2) silence resistance,
(3) stuffing resistance simultaneously.
**Status: CLOSED** — causality gate (≤5% pass on empty arm).

## A3 — Overconfidence / L1lift inflation  [HIGH — **RT-3b MITIGATED (bound+detector+duel); calibration REJECTED**]

**Mechanism**: thought-conditional decoder peakiness — inflate `lpA(y_C|z_A)` toward 0 while
leaving `lpA(y_C|∅)` unchanged. Purely analytic on stored pairs (no new rollouts). Raises
L1lift in mix=Λ2+L1lift; Λ2 (teacher channel) untouched.
**Mean-test RESULT (2026-08-03)**: strength≥0.25 lets I/II beat genesis on mean mix; V/VII
need ≥0.75. Uniform rescale of all lpA (incl. empty) still lets I edge genesis. Clip L1lift≥0
fails. Λ2-only ranking resists but re-opens I/II > genesis (ρ=+0.833, II#1).

**Calibration defense REJECTED (`scripts/rt3_calibration.py`, n=19)**: the proposed
`lpA(y_C|z_A) − max_k lpA(y_C|prior_k)` family (tested as `l1cal_zc`, baseline = teacher
thought `z_C`, the strongest self-scored baseline available offline) *fails on both axes*:
(a) ranking collapses ρ +0.856 → **+0.723**, genesis falls #1 → **#3** (II/I above it — the
foreign-thought baseline subtracts exactly the capability signal we want); (b) it does **not**
close the live threat — I/II still overtake genesis at strength 0.05 (mean) / 0.25 (duel),
because the attack scales the *gap* between z_A and the baseline, which calibration preserves.
Calibration only helps V/VII, which the duel already crushes (z≈−15). Wrong tool.

**What actually holds (three cheap, composable layers)**:
1. **Bound the per-turn term**: `clip0.1` = clip(L1lift, ±0.1) keeps ρ **+0.862** (≈ unbounded)
   and caps how much a single inflated turn can move S. `cap0.1`/`tanh0.1` similar.
2. **Two-sided serve-time calibration-ratio gate**: `r = mean|lpA(y_C|z_A)| / mean|lpA(y_C|∅)|`.
   Honest fleet band **[1.07, 3.56]** (genesis 1.07, I 1.33, II 1.17). RT-3b must drive `r` below
   ~1.0 to breach the duel → **out-of-band, detectable**. Same gate two-sidedly catches the
   mirror attack RT-3d (empty-pass *deflation*), whose `r` also leaves the band.
3. **3σ duel does the heavy lifting**: with a `r≥1.0` floor, the *optimal* per-pair water-filling
   attack (budget = how far logprobs can move before the detector trips) stays **under 3σ for all
   18 challengers** — l1lift z_max≈+2.15 (I), +2.07 (II); clip0.1 +2.91/+2.69. Only the tighter
   `cap0.05`/`clip0.05` bounds over-compress and let I (and II) breach — so **don't over-clip**.

**Realizability note**: strongest variant needs the miner's server to sharpen only the
z-conditioned pass. Bittensor intake pins HF weights + forbids custom modeling code
(`affine/model_store.py`: no `*.py`, no `auto_map`), and the eval box runs the miner's weights
under **our** vLLM, so a bespoke logit hook is not trivially deployable — the attack reduces to
serving temperature, i.e. the `uniform` column, which never breaches the duel for any king.
**Shipped (2026-08-03)**: `clip0.1` + `r∈[1.0, 4.0]` + duel δ=0.05 in `harness/score.py` and
`affine/affine/score.py` (`affine.toml` min_margin=0.05). Offline re-freeze: ungated ρ=+0.862@19.
**Status: SHIPPED + LIVE CONFIRMED (2026-08-03)** on affv-e7. Fresh 80-turn duels:
I margin=+0.0028 z=+0.20; II margin=−0.0005 z=−0.03 — both lose. Live r (genesis/I/II =
1.14–1.42) inside honest band. Calibration family remains rejected.

## A4 — King copying  [MEDIUM — **CLOSED RT-4**]

**Mechanism**: submit the current king's weights (or an epsilon-perturbation) — classic
king-of-the-hill parasitism.
**Defense**: identical model ⇒ identical S in expectation; dethronement requires beating the
king by δ ≥ k·SE(S_A − S_B) on paired turns. Paired-turn evaluation makes SE small and the
copier's expected margin exactly 0 ⇒ loses. Weight-space distance checks are a cheap extra.
**Experiment RT-4**: run the same model as king and challenger over 200 turns; measure the
paired-difference distribution to set δ (this doubles as the variance calibration for the
subnet's win margin).

**RESULT (2026-08-02, king-XLVI ×2, 80 turns)**: null |z|=1.48 < 3σ. **Status: CLOSED.**

## A5 — Teacher distillation shortcut (is it even an attack?)  [ACCEPTED OUTCOME]

**Mechanism**: miner maximally distills GLM within the 35B budget.
**Verdict**: this is the *intended* equilibrium — S is designed so the best response is
approximating the teacher's reasoning-to-action map. The subnet's value proposition is exactly
"pay for the best compressed GLM". No defense wanted; document as mechanism property.

## A6 — Dataset sniping / turn-distribution overfit  [MEDIUM — **MITIGATED RT-6 (ops; not a score gate)**]

**Mechanism**: D's turn distribution is public (mini-coder et al.); miner overfits to the
exact prefix distribution (e.g. memorizes trajectories, keys on artifact strings like
`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`) rather than learning the task.
**Experiment RT-6a (2026-08-03, `scripts/rt6_memorization.py`, offline)** — two candidate
*offline* detectors tested and **both fail**:
- **Per-turn concentration probe fails.** Hypothesis "memorizer = spiky per-turn S" is false: a
  synthetic memorizer (base=XLVI, memorized turns → genesis-level) *lowers* Gini 0.231→0.128 as
  it memorizes more, moving toward genesis, never into the honest tail (mean+3sd=0.265).
  Replacing low scores with high ones makes the distribution more uniform. Fleet top-1-turn
  share ≤0.9% for all 19 kings — no single-turn domination anywhere.
- **Artifact-string keying is moot.** The scaffold tokens (`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`,
  `SUBMIT_FINAL`, `FINAL_OUTPUT`, ```` ```bash ````, echo/cat/submit/import) are in **100%** of
  turns — no discriminative substring to selectively boost seen turns. Per-repo dispersion is
  capability-confounded (weak kings vary more; all share the same easy repos as best).
**Experiment RT-6b (2026-08-04, `scripts/rt6_temporal_holdout.py`)** — dated leave-repo-out
on the 20 commit-pinned repos in D (GH committer dates; early/late 10+10). Per-king mean S
recomputed from stored pairs:
| split | Spearman(S, swe) | n |
|---|---|---|
| all | +0.856 | 19 |
| early repos | +0.846 | 19 |
| late repos | +0.845 | 19 |
| early ↔ late | +0.947 | 19 |
Genesis is the only king with Δ(late−early)>0. No sniper-shaped late collapse.
**Conclusion**: score-distribution detectors cannot close A6; capability isomorphism is
*stable* under dated leave-repo-out on current D. Residual whole-corpus memorization is an
**ops** problem. Shipped mitigations: (1) fresh teacher `y_C` at duel time, (2) reveal-block
slice seeding, (3) `dataset.corpus_epoch` + `refresh_max_age_blocks` with mandatory
`weight_version_key` bump on corpus replace (audit fields on duel slice). Private pool
stays rejected (replayability).
**Status: MITIGATED (ops)** — not CLOSED as a score-function property; refresh remains the
operator duty.

## A7 — Variance gaming  [LOW — **CLOSED by design**]

**Mechanism**: near-deterministic miner shrinks its own sampling noise, exploiting a fixed
win margin δ.
**Defense**: δ set from the *measured paired variance of the current pair*, not a constant
(RT-4 gives the estimator). Deterministic miners also hurt themselves on diversity-sensitive
turns.

## A8 — Tokenizer games  [LOW — **CLOSED by design**]

**Mechanism**: exotic tokenizer to shift per-token probabilities.
**Defense already in place**: all comparisons are per *byte* (BTR lemma), injection/forcing is
byte-exact through each model's own template, span located by offset mapping. Residual risk:
tokenizer with huge action-fragment tokens changes the granularity of the forced span but not
the byte-normalized total.

## A10 — Soft-idents padding (surface regularizer Goodhart)  [HIGH — **CLOSED (abandoned; RT-soft-pad confirmed)**]

**Mechanism**: production briefly considered softΛ2 = Λ2 − α·relu(T−n_idents(z))/T to
fix the king-I brevity false positive. A miner appends `# id1 id2 …` to z. Offline
(logprobs frozen): junk pad restores I>genesis (win 64%); even prefix-grounded idents
(idents ∩ prefix) are pad-gameable by scraping the prompt (win 68%).
**Verdict**: any published function of raw z text is Goodhartable under adversarial miners.
**Defense / replacement**: drop soft-idents from S*. Rank with teacher-channel mix
Λ2 + 1.0·L1lift. On 15 kings: Spearman +0.844 ungated / **+0.912** hybrid (frozen);
genesis #1; RT-2c para win ≈21% under mix before bank gate. Keep bank + causality + 3σ duel.
**Status: CLOSED (abandoned)** — soft-idents removed from production S\*.

## A11 — Short-style ranking FP (I, II)  [MEDIUM — **CLOSED by δ=0.05 effect floor + live RT-3b**]

**Mechanism**: concise, non-leaking thoughts (|z|≈180–200, low leakage) rank high on Λ2/mix
despite weak swe-rebench (I 38.4, II 37.2 vs genesis 58.2). **Bank gate does not reject**
I or II (bank_frac 0.30/0.30 — not prior-stuffers). w=1.0 mix restores genesis #1 on
n=15 E-KINGS ranking, but 80-turn duel slices can still flip paired mean S.
**Defense**: 3σ duel rejects crowning (re-scored under current mix w=1.0: z=**0.59** for I,
z=**1.72** for II — both below 3σ; the stored z=1.30 for I was a superseded term).
**Fragility quantified (2026-08-03, `scripts/a11_bootstrap.py`, bootstrap B=20000)**: a pure
z>3 test fires for *any* persistent margin>0 at large enough n. II's margin is small but
positive (+0.034, Cohen d=0.19), so it would cross 3σ at **n\*≈244** paired turns; I at ~2071.
**clip0.1 makes this worse** (shrinks variance → I z 0.59→0.99, II n\* 244→189) — it defends
RT-3b but not A11. **Fix = effect-size floor**: short-style FPs (II +0.034, I +0.009, d<0.2)
are cleanly separated from correct rejections (XCIX −0.38, LI −0.60, VII −0.65, d<−0.9). A floor
**δ=0.05** (~8% of the −0.036…−0.63 king-S range) blocks I/II at any n while a genuine capability
jump (margin ≫ δ) still dethrones; it also caps small-margin RT-3b duel inflations.
**Status: CLOSED** — `min_margin=0.05` shipped in S\* v2; live I/II duels under clip0.1
z=+0.20/−0.03 (neither crowns).

## Live duel board (frozen 2026-08-03)

Genesis king vs challengers, 80 paired turns, mix @ w=1.0, γ=0.30, γ_bank=0.08.
Source: `results/duel_genesis_*.json`.

| challenger | margin (S_c − S_k) | z | bank_frac | valid | dethrone? | verdict |
|---|---|---|---|---|---|---|
| I | +0.009 | 0.59 | 0.503 | ✓ | no | **fix: effect-floor** — re-scored w=1.0 (was 1.30 old term); tiny margin |
| II | +0.034 | 1.72 | 0.444 | ✓ | no | **fix: effect-floor** — n*≈244 under pure 3σ; δ=0.05 blocks |
| VII | −0.649 | −10.53 | 0.128 | ✓ | no | **CLOSED** — large gap; genesis retains |
| XCIX | −0.382 | −8.60 | 0.153 | ✓ | no | **CLOSED** — large gap; genesis retains |
| LI | — | — | 0.078 | ✗ | no | **CLOSED** — bank-invalid; cannot challenge |

Hybrid freeze: Spearman +0.912 (n=10), rejects XI/XLV/XLVI/CI/LI — confirmed by
`bank_w2_fullz` late-king rescoring (`results/hybrid_freeze_meta.json`).

## A12 / RT-7 — Isomorphism inverts on the live panel  [SEVERE — **OPEN, not defended**]

**The claim this breaks.** MOTIVATION says S stays benchmark-isomorphic *under
adversarial pressure*. Every panel behind the +0.758 freeze is Albedo kings —
models optimised against a **GLM judge**, not against S. They are adversarial to
SN97, not to us. The live SN120 board is the first genuinely adversarial panel,
because every submission on it was made by someone maximising S.

**Result (validator's own published artifacts, n=29 live challengers):**

| statistic | value | p (permutation) |
|---|---|---|
| Spearman(duel margin, swe_lite) | **−0.421** | 0.024 |
| Spearman(S absolute, swe_lite) | −0.371 | 0.049 |
| freeze, Albedo panel (n=30) | **+0.758** | — |

The sign flips. Corroborating, independent of the correlation:

- **Every model crowned by S scores 0.00 on swe** — reign-2 kevin954, reign-3
  TalentPigs, reign-4 Tok331102, all 0/25. Only reign-0 genesis (0.20) is
  non-zero, and genesis was *seeded*, never won a duel. 37% of the field sits at
  zero, so 3-of-3 is p=0.052 by itself.
- **The untouched base model wins the benchmark.** `Qwen/Qwen3.6-35B-A3B` scores
  **0.24**, the best of 51 benched models. 400+ submissions of "improvement" and
  none beats doing nothing.
- **Within one miner, the crowned checkpoint is the broken one.** Tok331102 `af5`
  scores swe 0.16 and *lost* its duel (S=−0.014); the same miner's `af10` scores
  **0.00** and *took the crown* (S=+0.0446). Optimising S destroyed swe inside a
  single lineage — Goodhart with the confounds held fixed.
- **Mechanism, measured independently.** An external red-team run screened raw
  published models against the king over n=80 with all gates clear: raw genesis
  m=**−0.05489** (z=−6.05, r=0.977, base_x=1.009), and eight others −0.006…−0.087.
  Every one loses through **Λ2**, not through a gate (λ2_c −0.017…−0.029 vs king
  +0.005). Λ2 rewards *thoughts that help the teacher*, and the incumbent maxes
  that by construction, so Λ2 behaves as a **similarity-to-incumbent term**, not
  a capability term. 45 structurally distinct families all failed this way.

**Why the gates do not catch it.** Every gate is a *validity* check on the pair
(causality, leakage, bank, r, baseline band). None of them asks whether the
winner can write code. A model can be perfectly gate-valid, crown, and resolve
0/25.

**Status: OPEN.** No defense shipped. This is the SN97 pathology the subnet was
built to fix, reproduced by a different mechanism: SN97 was genesis 58.2 → kings
26–38, SN120 is genesis 0.20 → kings 0.00, i.e. the kings are now at the floor.

Artifacts: `research/results/rt7_live_isomorphism.{json,txt}`,
`research/scripts/rt7_live_isomorphism.py` (re-runs from live published data).

**Caveats.** swe_rebench_lite is 25 tasks (0.04 granularity, floor-heavy) where
the freeze used 500; `history.json` exposes only the last 100 events, capping n
near 30; margin is paired within a duel but still spans different kings/slices.
The direction and the three corroborating facts do not depend on the instrument.

---

## Status summary (2026-08-03)

| ID | attack | status | defense |
|---|---|---|---|
| **RT-7 / A12** | **isomorphism inverts live** | **OPEN** | **none — ρ=−0.42 (p=0.024); all 3 S-kings swe=0.00; base model best of 51** |
| RT-1 / A1 | fixed payloads | **CLOSED** | lose to genesis; empty → causality gate |
| RT-2 / A2 | exact stuffing | **CLOSED** | leakage gate (0% pass) |
| RT-2 / A9 | silence | **CLOSED** | causality gate (≤5% pass) |
| RT-2c / A2c | paraphrase stuffing | **CLOSED** | bank gate (frac=0); mix w=1 ~21% win |
| RT-soft-pad / A10 | soft-idents pad | **CLOSED** | abandoned; use mix w=1.0 |
| RT-4 / A4 | king copier | **CLOSED** | 3σ duel (|z|=1.48) |
| A11 / live I | short-style FP | **CLOSED** | δ=0.05 floor + live RT-3b z=0.20 |
| A11 / live II | short-style FP | **CLOSED** | δ=0.05 floor + live RT-3b z=−0.03 |
| Live duel VII | large-gap challenger | **CLOSED** | z=−10.53; genesis retains |
| Live duel XCIX | large-gap challenger | **CLOSED** | z=−8.60; genesis retains |
| Live duel LI | bank-invalid challenger | **CLOSED** | frac 0.078 < γ_bank; cannot challenge |
| bank_w2_fullz | late-king bank rescoring | **CLOSED** | confirms XI/XLV/XLVI/CI/LI rejects; hybrid +0.912 frozen |
| RT-3 / A3 | L1lift inflation | **CLOSED (live)** | clip0.1 + r∈[1,4] + δ=0.05; live I/II z=0.20/−0.03 |
| RT-3d / A3d | empty-pass deflation | **MITIGATED offline** | two-sided r gate (mirror of RT-3b) |
| RT-6 / A6 | dataset sniping | **MITIGATED (ops)** | detectors fail; dated leave-repo ρ≈+0.85; corpus_epoch refresh |
| judge baseline | same-turns apples-to-apples | **S wins** | GLM-5.2 +0.09/+0.345 vs S +0.827; gpt-4.1-mini anti-correlates |
| A5 | teacher distillation | **ACCEPTED** | intended equilibrium |
| A7, A8 | variance / tokenizer | **CLOSED** | paired δ; byte-normalized |

## Priority queue (updated 2026-08-03)

1. ~~RT-2 exact stuffer~~ **CLOSED** — leakage gate (0% pass).
2. ~~A9 silence~~ **CLOSED** — causality gate (≤5% pass).
3. ~~RT-1 fixed payloads~~ **CLOSED** — lose to genesis; empty needs gate.
4. ~~RT-2c paraphrase~~ **CLOSED** — bank gate; mix@w=1 win≈21% vs G.
5. ~~RT-4 copier~~ **CLOSED** — null |z|=1.48 at n=80.
6. ~~RT-soft-pad~~ **CLOSED** — soft-idents abandoned.
7. ~~Live duels VII / XCIX / LI~~ **CLOSED** — z=−10.53 / −8.60 / bank-invalid.
8. ~~bank_w2_fullz~~ **CLOSED** — late-king bank rejects confirmed; hybrid +0.912 frozen.
9. **A11 short-style I/II** — **fix identified**: pure 3σ fires for any margin>0 at large n
   (II n*≈244); clip0.1 worsens it. Ship **effect-size floor δ=0.05** (dethrone iff z>3 AND
   margin>δ) — separates FPs (II +0.034) from real rejections (≤−0.38). `min_margin`=0.05.
10. ~~RT-3 overconfidence~~ **CLOSED live** — S* v2 shipped; live I/II duels z=+0.20/−0.03;
    live r∈[1.14,1.42] in honest band. Calibration family rejected.
11. **RT-6 dataset sniping** — **MITIGATED (ops)**: score detectors fail; dated leave-repo-out
    STABLE (ρ_late=+0.845); `corpus_epoch` refresh + fresh y_C + reveal-hash slice shipped.
12. **Judge baseline** — **S wins on same inputs**: production GLM-5.2 +0.09 (holistic) / +0.345
    (anchored, reproduces prod +0.31) vs S +0.827 on identical 11 kings × 50 turns; gpt-4.1-mini
    anti-correlates. `scripts/judge_same_turns.py`.
