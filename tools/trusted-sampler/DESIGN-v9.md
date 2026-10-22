# SN120 v9 "measured impact": miner-payment design spec

*Draft for Jacob Steeves (subnet owner), to review and share with Allan. Written Mon Oct 19, 2026, ~8:40 PM CR (UTC-6). Miner data appears only as aggregates; no UIDs or keys are listed.*

**Summary.** Today SN120 miners are paid for volume of valid rollout groups. They all run the same honest code, so nothing in their data differs, and no scorer we tried finds a stable per-miner signal. v9 changes what we pay for. A miner earns extra weight only when its data *measurably moves the model further than honest sampling would*. The validator estimates how much training on each small batch would raise a reference objective, using a gradient dot product. The reference is drawn from the full env by a public random value (a future block hash) revealed only after the commit deadline, so it is unpredictable beforehand and publicly verifiable afterwards. Each batch is tested against that epoch's honest-data null. The results are combined over a window of batches per miner with a calibrated test. Only miners who clear a multiple-testing-corrected gate get a share, paid in proportion to their estimated improvement above the honest baseline. That share is blended with the existing weights behind a signed kill switch. Offline, the method is strong on the target case: per 3-group batch, one group with genuinely new correct solutions is detected at ≥80% power after about 6 batches at α=0.01 (8–10 on the hardest epoch), with 0–5% false positives. At the multiple-testing-corrected α=0.01/210 this rises to 11–12 batches, and 80% is not reached within 12 on the hardest epoch. Its blind spots are just as clear. It does not tell honest data from fake, garbage, easy or stale data (the verifier handles labels). The adversarial run (§7, RC-adv) found:
- **One blocking hole.** The live epoch-number seed lets miners rebuild the reference and select on it.
- **Two economic holes.** Paraphrased copies and replays are paid again. Selecting long or high-gradient honest groups pays like real beyond-policy data, but it also produces real held-out gains: this is curation, and it needs a policy decision.
- **No Goodhart break.** Every high-scoring class raised held-out J in real scratch steps.

At α=0.01/210 the per-miner gate needs ≈11–12 batches for 80% power on one new group per batch. Rollout is shadow → small blend → ramp. The gates are calibration, canary power, linearization audits and a held-out drift alarm.

**Tags used below.** **[M]** measured tonight (source cited). **[D]** simple arithmetic on measured numbers. **[P]** a design proposal or chosen value, not evidence. **[U]** an assumption that has not been tested.
Sources: `REPORT-claims.md` sections *Real rollouts*, *Planted batches*, *Scoring rules*, *Detection threshold*, *Per-miner aggregation* and *v9 adversarial + scalability* (cited as RC-real, RC-plant, RC-rules, RC-det, RC-agg, RC-adv). `REPORT.md` (the trusted-sampler report, cited as TS). `/workspace/sn120-impact/` (the live release, cited as REL).

---

## 1. What we know going in

| finding | number | tag |
|---|---|---|
| Real miners have no stable per-UID signal | cross-epoch UID Spearman −0.12…+0.06 (e62–e65); release gate −0.07…+0.07 (e62–e66) | [M] RC-real, REL |
| Real labels are honest | 0 / 56,592 rollouts disagree with our verifier (e62–e65), 0 / 15,048 (e66) | [M] RC-real |
| Real batches are all on-policy | 100% built on the epoch's base checkpoint (admission enforces it) | [M] RC-real |
| Pass-rate deltas cannot price a single batch | full 256-group step +0.002 ± 0.005; one batch ≈1e-4; would need ~10⁷ paired tasks per batch | [M] RC (ground truth) |
| Volume regression on pass-rate deltas cannot price miners | 0/214 UIDs separable, empirical-Bayes heterogeneity at its floor | [M] TS §4 |
| Gradient-dot sees *large* quality differences | beyond-policy batches score +6.9 / +8.4 SD vs the real population, AUC 1.00 | [M] RC-plant |
| …and nothing smaller | random rewards, 30% inverted, garbage-as-correct, easy-only, hard-only, stale, good-balanced all fall inside the honest spread; 100% inverted is caught only weakly (AUC 0.06 / 0.19) | [M] RC-plant |
| First-order score matches a real update | per-chunk corr 0.76–0.93, slope 0.87–1.12 on held-out tasks; tops beat bottoms 8/8. Exception: garbage, corr 0.24 | [M] RC scratch-step, RC-plant |
| Max-group beats batch mean; cosine is worse | 9-group batch, k=1 new group: raw 6–12%, max-group 21–54%, cosine 0–8% detection | [M] RC-det |
| Small batches help | k=1 power, max-group: B=3 .42–.54, B=9 .27–.42, B=36 .14–.34 | [M] RC-det |
| Per-miner aggregation reaches 80% power | 3-group batches, k=1: Stouffer ≈6 batches (e62/e64), ≈10 (e66). FP 0–5% Stouffer, 0–3% binomial | [M] RC-agg (simulation on stored dots) |
| The naive any-hit rule is gameable | splitting 36 groups into 1-group batches flags honest miners 20–61% of the time; binomial ≤0.6% | [M] RC-agg |
| Real UIDs under the calibrated tests | binomial 0.9%, Stouffer 1.4% flagged (α=0.01), vs any-hit 22%; hit histogram fits independent batches at q≈0.06 | [M] RC-agg |

**What this means for the design.** Honest miners have nothing to tell apart today. The useful thing the score can detect is *new correct solutions the policy almost never finds*. v9 should pay for that, and for any other measured gain, and fall back to the current rule otherwise.

## 2. Goals and non-goals

**Goals**
1. **Pay for measured improvement, not for effort or for a recipe (the bitter lesson).** We do not hand-score difficulty, subject, length or "balance". Miners may use search, bigger compute, tools or other models. If it measurably helps, it is paid. Subject claims were tested and add no consistent signal; they are gameable on regenerated data (RC verdict). So v9 has no claim field.
2. **Calibrated by construction.** Each epoch, every threshold comes from that epoch's honest null, not from a fixed constant. Sensitivity varies about 2× across epochs (e66 is about half of e64; [M] RC-det).
3. **Few hand-tuned knobs.** The full list is: blend `b`, window `W`, family-wise error `α`, batch size (3 groups), and the per-UID batch cap. Everything else is derived.
4. **Fail safe.** If anything is missing or unhealthy, the writer returns exactly the current-rule vector. That is how REL already behaves.
5. **Keep training useful.** Scoring must not throw away submitted data. The trainer stays the live v8.2.4 PPO-style trainer.

**Non-goals**
- **Ranking honest miners against each other.** There is no signal for it ([M]), and v9 does not try.
- **Detecting fake, garbage or easy data with the score.** It cannot ([M] RC-plant). Label fraud belongs to the verifier.
- **Pricing every batch in exact pass-rate points.** That is infeasible ([M]). The score is a first-order proxy, audited against real steps.
- **Policing near-duplicate text.** Jacob prefers privacy-for-N-epochs over dedup policing. Only exact-duplicate dedup is in scope.

**Jacob's "bounded by the reference" concern.** The reference is not a gradient for miners to imitate. It is the gradient of a *measurement*: J = mean over reference tasks of [mean log-prob per token of correct samples − mean of wrong samples]. A batch can exceed anything the validator's own sampling produced. Beyond-policy groups on tasks the base solves ≤1/8 of the time, which are never in the reference chunks, still raise J by 7–8 SD ([M] RC-plant). The real limit is coverage: J is only defined on tasks where the base passes 20–80%, so gains on never-solved tasks count only through transfer. See open question Q4.

## 3. Miner protocol

| item | rule | tag |
|---|---|---|
| Group | one task index from the epoch's offered pool, 8 rollouts (prompt tokens, completion tokens ≤8,192, claimed classification and reward), with the existing group contract (≥1 correct, ≥1 non-correct) | [M] these are today's fields; [P] keep |
| Batch | **exactly 3 groups** (24 rollouts). A fixed size means one null per epoch and no gain from splitting | [P], motivated by [M] RC-agg |
| Batches per UID per epoch | up to **3** (9 groups, the same ceiling as today's ≤9-group commitment, so validator load is unchanged) | [P] |
| Built-on | must name the epoch's base checkpoint; anything else is rejected | [M] enforced today |
| Commit | the content hash of each batch is committed through the existing commitment path before the epoch's submission deadline. The bytes go to the miner's private slot | [P] reuse of the existing mechanism |
| Dedup / first submitter | (1) Exact key = hash(task index, completion tokens) per rollout. Only the **earliest commitment** (block number, then extrinsic index) gets credit; later copies score 0; a batch that is ≥2/3 copies is void. (2) **Task-level first-solver credit**: within the window, beyond-honest credit on task i goes to the earliest commitment. Later batches on task i are scored but paid only for the excess over it. (3) **Cross-history ledger** of (task, completion hash) and of paid tasks | [P]. [M] RC-adv: paraphrased copies (base rewrites from a hint; 5-gram Jaccard 0.07) evade (1) and score like the originals (gated 0.77, V 35 vs BP 0.83, V 37); prefix-50% copies 0.88 |

| Privacy | batches stay private for N epochs, then may be released. Replay of released data is allowed for *training*, but it is **not paid again** (ledger) | [P]. [M] RC-adv: e62 beyond-policy groups replayed at e64 still score z 4.4 (gated 0.94, V 47, more than fresh BP), so without a ledger the same data is paid every epoch until the policy learns it |

## 4. Validator scoring pipeline (per sealed epoch E, off the writer's critical path)

1. **Fetch.** Read-only fetch of E's documents. REL already does this (`fetch_epoch.py`, list and get only).
2. **Verify.** Check each batch's hash against its commitment, its built-on checkpoint, its format and the 3-group size. **Re-grade every rollout** with the verifier replica. Any mismatch voids the batch. (Proposal [P]: also void the UID's batches for the window.) This is the only label-fraud defense: even 100% inverted labels move the score by only ~2 SD ([M] RC-real flips). Then apply dedup.
3. **Reference: a public post-commit random draw from the full envs.**
   - **Seed.** A public, unpredictable value revealed only after the miner commit deadline, e.g. the hash of block (deadline + d). It is unpredictable before commits close and verifiable by anyone afterwards. There is no private set and no secret salt.
   - **Draw.** About 1,400 tasks are drawn from the full env pool (affine_math: 6,746 offered tasks), stratified by subject (or env). Each gets 8 base samples, seeded from the same value.
   - **Band filter.** Keep tasks with 2–6 of 8 correct, computed *after* the draw: [M] 330–350 per epoch. Keeping the band is [M] RC-adv: adding the 1/8 and 7/8 tasks cut BP mean z from 3.85 to 2.07 and gated power from 0.73 to 0.58.
   - **Split.** The kept tasks form 4-task chunks in two disjoint halves: [M] 80–84 chunks.
   - **Leave-task-out.** When scoring a group, drop the chunks that contain the group's own task. It costs nothing and changes nothing on honest data ([M] RC-adv). Submitted tasks are *not* excluded from the draw, because that would let miners steer the reference.
   - **Fix needed (blocking).** The live code seeds the draw with `H('ref', E)` and `H('chunks', E)` (`gen_ref.py`, `grad_claims.py`), which anyone can rebuild. Selecting the top 3 of 32 honest groups on that exact reference is gated 0.88–0.96, with V 71–107 (2–15× a legit miner) ([M] RC-adv). This **must be replaced by the post-commit draw before any blend > 0**.
   - **Critical path.** The reference can only be generated after the deadline: ≈8 min on 8 B200, ≈1 h on 1.
4. **Gradients.**
   - **Subspace.** Reference chunk gradients G_k of J are taken at the base in the MLP weights of the last 4 of 32 layers (604M parameters).
   - **Per-rollout dots.** One backward pass per rollout, then dot products with every G_k ([M] REL).
   - **Group score.** s_g = Σ_j (r_j − r̄) · d_j / (8192·N), with the live overlong reward. It is averaged over all subject chunks in both halves (the "raw / all-subjects" estimator).
5. **Honest null (calibration, per epoch).**
   - **Tested version.** The distribution of the max over 3 group scores, built from honest groups that are *out of sample*: a CAL half that is never used to score the same groups ([M] method RC-agg, 200k resampled batches).
   - **Proposed primary null [P][U].** Groups the validator samples itself from the base on random non-reference pool tasks, with the miner harness. A real-population null still works today, but it would absorb real improvers once they appear. That would cost power, not false positives.
   - **Placebo checks, kept from REL.** The label-shuffle null mean/SD must stay near 0: [M] 0.035–0.037 measured, and REL's limit is 0.5. A zero step must give exactly 0.
6. **Per-batch statistic.** p_b = P_null(max_g s_g ≥ observed). A batch is a "hit" when p_b < q, with q = 0.03 at S=3 ([M] RC-agg).
7. **Per-miner test over the window** (the last W epochs, up to 3W batches).
   - **Stouffer score:** Z_i = Σ_b Φ⁻¹(1−p_b)/√B_i.
   - **Gate:** a binomial test P(Binom(B_i, q) ≥ hits_i) < α/N_UID, plus Z_i above the matching Bonferroni quantile.
   - **Never use** any-hit or summed raw z ([M] gameable).
   - **Multiple testing.** N_UID is the number of scored UIDs, [M] 205–218 per epoch, so α = 0.01/210.
   - **Batches for 80% power at k=1** ([M] RC-adv, e62 / e64 / e66): Stouffer + binomial 12 / 11 / >12; Stouffer alone 9 / 8 / >12; BH q=0.05 6 / 5 / >12 (0.78 at 12).
   - **At k=2:** 6 / 6 / 12 (v9 gate), 3 / 3 / 5 (BH).
   - **False positives.** Honest FP at the v9 gate is ≈0. BH's realized FDP was 0.13–0.16 on e62 (above q) because of null-tail miscalibration, so BH is only usable with a conservative pooled null (Q2).
8. **Payment value.**
   - **Formula [P].** For gated miners, V_i = max(0, Σ_b Σ_g (s_g − m_E)), where m_E is the honest-null median group score. This is the estimated ΔJ *above honest baseline*. Honest data has positive mean ΔJ ([M]: 16–39 of the 205–218 real batches per epoch have z>2 vs the shuffle null, because honest labels raise J). Paying for baseline would just pay for volume again, which the current rule already does.
   - **Ungated miners.** V_i = 0.
   - **Replaces REL's rule.** REL's current rule (EMA of batch score where z>2 vs the *shuffle* null) would pay noise to 7–19% of honest real batches per epoch ([D] from RC-real z>2 counts), so v9 replaces it.
9. **Weights.**
   - **Blend.** impact_i = V_i / Σ V. Final share = (1−b)·current_i + b·impact_i, which is REL's existing `blend_points` with owner exclusion.
   - **Fallback.** If no miner is gated, impact is empty and the writer returns the current vector unchanged.
   - **Window instead of EMA.** v9 uses a fixed sliding window of W epochs. REL uses an EMA (`ema_epochs` 6), but the test is calibrated for an unweighted count of batches. If an EMA is wanted, use a weighted Stouffer [P].
   - **Proposed default:** W = 4 at 3 batches per epoch, i.e. 12 batches. That reaches ≈80% k=1 power on e62/e64-like epochs at α=0.01/210 ([M] RC-adv). e66-like epochs reach 80% only at k≥2.
10. **Sign and publish** with the dedicated scorer key: aggregates per UID only (B_i, hits, Z_i, gate, V_i), placebo stats, and the draw seed (block number and hash) so anyone can rebuild the reference. This matches REL's signed score-file format, with new fields.

## 5. Training: what the trainer consumes

**Today [M].** About 1,700–1,900 groups are admitted per epoch and 256 are trained. So about 86% of submitted data is never trained on; this is Jacob's waste concern.

**v9 proposal [P][U]:**
- **Train on hits.** The trained set is every group from batches that were hits (p_b < q) in epoch E, filled up to 256 by uniform random draw from the remaining verified groups. That way the data we pay for is the data we train on.
- **Order of operations.** Scoring is done at the base before training, so the choice of trained set cannot feed back into that epoch's scores.
- **Hit cap.** The hit share should be capped, e.g. ≤50% of the step [P]. Otherwise a score exploit could take over the whole step, and the drift alarm (§6) needs a random control fraction.
- **Logging.** Log, as an aggregate, how many trained groups came from hits.
- **"Not wasted" also means measured.** Every verified batch is scored even if it is not trained on. More, smaller steps (Jacob's preference) would train more of the pool. That trade-off belongs to Allan (Q6).
- **Caveat.** The score assumes plain SGD at the base checkpoint, without Adam preconditioning, KL or clipping. In the live step a batch is mixed with 255 other groups, and GRPO groups are not additive ([M] TS §4 caveat). Section 6 checks whether this matters.

## 6. Held-out set and drift alarm (Goodhart guard)

- **Held-out tasks [P].**
  - **Reserve.** A fixed slice that is **never offered to miners and never trained on**. It must sit outside the training pool, because the reference is drawn from the same pool the trainer trains on, so J can rise from pool memorization ([M] RC-adv: affine_math is a fixed public 7,496-task snapshot with answers included). Candidates: the 750 snapshot tasks the live index does not offer, if they are confirmed never used [U]; later, other envs that are never scored. Membership is public; held-out-ness comes from never training on it, not from secrecy.
  - **Trusted sampler.** The existing trusted sampler measures real paired pass-rate deltas (CRN, m=8) on this slice every step.
- **Alarm 1: realized vs predicted [P].**
  - **Precision.** Per step [M]: paired SE ≈0.0046 at ~900 tasks. Over 6 steps the precision-weighted SE was 0.0018 ([M] TS §1), so a sustained drop of roughly −0.004 per step is visible within ~6 steps ([D]).
  - **Trip rule.** Trip if the 6-step pooled held-out delta is < −2 SE, **or** if the pooled delta of steps with many hit groups is significantly below steps with few. Hit share is random-filled, so this works as a weak natural experiment [U].
- **Alarm 2: linearization audit [P].**
  - **What.** Each epoch, take the top-3 paid batches plus 3 random batches and run a real scratch step on an fp32 copy of the subspace, then measure J on the held-out half. Cost: about 10 GPU-min per batch ([M] RC scratch-step).
  - **Trip rule.** Trip if corr(real, predicted) < 0.5 or slope is outside [0.5, 1.5] for a paid batch. The measured honest range is 0.76–0.93 / 0.87–1.12. Garbage fell to corr 0.24 ([M]), which is exactly the failure this catches.
- **Alarm 3: calibration [P].** Trip if the real-population per-UID flag rate at α=0.01 exceeds 3% (measured 0.9–1.4%), or if the placebo |mean/SD| is > 0.5.
- **On any trip:** blend → 0 automatically (a new kill reason in `measured_impact_blend.py`) and a human review.

## 7. Threat model

**Setup** ([M] RC-adv, *v9 adversarial + scalability*):
- Stored dots, e62/e64/e66, with 5 CAL/SIM splits.
- v9 gate at α=0.01/210, B=10 three-group batches.
- "Gated" = fraction of simulated miners passing. V = payment in honest-SD units.
- Comparison point: a legit beyond-policy (BP) miner, k=1: gated 0.59 / 0.83 / 0.20, V 21 / 38 / 7.
- Real held-out effect = 2 real scratch steps, ΔJ×1e4 on the held-out half. Honest batches: −1.0.

| threat | attack | v9 payoff (gated; V) | real held-out effect | defense in v9 | status |
|---|---|---|---|---|---|
| Label fraud | wrong answers marked correct, random labels | score alone misses it | — | verifier regrade voids the batch | **[M]** 0 wrong labels in 71k real rollouts |
| Garbage / easy / hard-only padding | cheap filler | easy ≥75% pass: 0.00; garbage-as-wrong: ≈0 (or void, no reward variance) | easy −1.2 (= honest) | not paid: inside the null | **[M]** |
| Reference leakage, **live epoch-number seed** | rebuild the exact reference from `H('ref',E)`, `H('chunks',E)`, then submit the top 3 of 32 honest groups | **x32 0.90 / 0.88 / 0.96, V 71–107. x64 1.00, V 102–168** | selection transfers (+2.7) but is overpaid ~5× per unit gain | **public post-commit random draw from full envs** (§4.3) | **[M] blocking, must fix** |
| Proxy cherry-picking under the post-commit draw | own same-distribution draw used as proxy; submit top 3 of 3n | x8 ≤0.08. x32 0.27–0.54 (V 11–24). x64 0.58–0.77 (V 23–52) | +2.7 (real) | post-commit draw (cuts V ~3–6×); residual = variance lottery below | **[M]** residual accepted or handled by curation policy |
| Subject / difficulty-mix guessing | target the full-env mix | ≤0.05, V ≤1 | — | none needed | **[M]** |
| Variance lottery (no reference needed) | top 3 of 3n honest groups by length or gradient norm | length x32 0.45 / 0.70 / 0.84 (V 14–59); gnorm x64 0.76–0.99 | **length +5.8 (real gain)** | Rejected: cosine score (kills BP: 0 / 0 / 0), grad-norm-conditional null (BP 0.25 / 0.21 / 0.00), studentized, split-half, replication. **Open policy (Q11):** pay as curation, or curate validator-side and pay only above that | **[M]**, decision needed |
| Near-duplicate copies | hint-paraphrase / prefix copy of another miner's BP solutions | hint 0.77, V 35; prefix50 0.88, V 43 | +4.0 (real) | first-solver credit per task plus ledger (§3); exact hash misses them | **[M]**; defense **[P][U]** |
| Replay of old / released data | e62 BP groups re-submitted at e64 | **0.94, V 47** | real (BP class) | cross-history ledger; train-on-hits makes it decay | **[M]**; defense **[P][U]** |
| Sybil splitting across UIDs | K BP groups over h hotkeys | total V (K=20, e62): h=1 60, 2 50, 3 19, 5 4, 10 0.3 | — | concave gate: splitting loses. Noise FP per hotkey ≤7e-4 | **[M]** |
| Teacher distillation | stronger off-policy model (066dfb10) writes solutions to hard tasks | **0.79, V 37 (= BP)** | **+4.6 (real)** | allowed (bitter lesson); audit plus drift alarm (Q10) | **[M]** |
| Task-pool memorization / answer-hinted rationalization | pool is a fixed public snapshot with answers; miners can cover all 6,746 tasks | in-sample groups (task in a scored chunk, 2.7–3.6%): mean z −0.06 / −0.05 / +0.58, unchanged leave-task-out | hint analog +4.0 | leave-task-out scoring; first-solver credit; held-out set outside the trained pool (§6). Do not rely on pool secrecy | **[M]** overlap; gold-answer rationalization **[U]** (hint analog measured) |
| Gradient inflation | huge-gradient sequences | garbage did not inflate; selected high-gnorm honest groups do score (row above) | real gain | scratch audit; curation policy | **[M]** |
| Optimizer mismatch | score assumes SGD; trainer uses Adam | — | sign-update (Adam first-step proxy): same ordering, smaller margins (honest −0.4, length +0.5, BP +0.5 vs SGD −1.2 / +1.7 / +3.4); per-chunk corr 0.43–0.62 | run the linearization audit in the trainer's update form; consider a preconditioned dot (Q3) | **[M]** partial (8 batches, 1 step) |
| Validator compute DoS | flood or max-length rollouts | — | — | caps; scorer off the writer path; REL auto-kill | partly tested; load at cap estimated in §8 |
| Scorer compromise | forged or stale scores | — | — | pinned ed25519 key, epoch/base binding, max age | **live** in REL (shadow) |

## 8. Scalability

**Model** ([D] from [M] throughputs, RC-adv):
- Per-rollout gradient dots at **8 rollouts/s per B200**; ×2.7 for all-long rollouts in parentheses.
- Post-commit reference ≈ **1 GPU-h** (≈8 min on 8 GPUs). It is on the critical path, because it can only start after the deadline.
- Calibration and gate on CPU. Rollouts = miners × batches × 24.

| miners × batches/epoch | rollouts | GPU-h incl. reference | scoring wall clock, 1 B200 | scoring wall clock, 8 B200 | 1 B200 per epoch (3.6–4 h)? |
|---|---|---|---|---|---|
| 100 × 1 | 2.4k | 1.1 | 65 min (75) | 10 min | yes |
| 100 × 3 | 7.2k | 1.25 | 75 min (100) | 12 min (15) | yes |
| 100 × 10 | 24k | 1.8 | 1.8 h (3.2) | 16 min (27) | yes |
| 256 × 1 | 6.1k | 1.2 | 73 min (95) | 12 min (14) | yes |
| 256 × 3 | 18.4k | 1.6 | 1.6 h (2.7) | 15 min (23) | yes |
| 256 × 10 | 61k | 3.1 (6.8) | 3.1 h (6.8) | 26 min (54) | borderline / no |
| 1000 × 1 | 24k | 1.8 | 1.8 h (3.2) | 16 min (27) | yes |
| 1000 × 3 | 72k | 3.5 (7.8) | 3.5 h (7.8) | 29 min (61) | borderline / no |
| 1000 × 10 | 240k | 9.3 (23.5) | 9.3 h (23.5) | 72 min (3 h) | no: 3–6 B200 |

**Notes.**
- **Audits.** The linearization audit adds ≈1 GPU-h per epoch (6 batches), off the critical path.
- **Cost** at $8.89/GPU-h (the pod rate): 256 × 3 ≈ $14 per epoch; 1000 × 10 ≈ $83–210 per epoch.
- **Bottlenecks:**
  - The fixed reference dominates up to ~256 × 3. Beyond that, per-rollout backward passes dominate.
  - Production needs only 2–3 combined reference vectors (1.2 GB each) instead of the 101 GB per-chunk matrix.
  - `causal_conv1d` is not installed and sequences are not packed; an estimated 3–5× speedup is available [U].
- **Beyond affine_math [U]:**
  - reference generation and chunk gradients scale per env;
  - nulls must be per env (length and gradient-norm distributions differ);
  - non-math verifiers add critical-path latency;
  - per-env testing multiplies N in α/N;
  - for thousands of envs, use the post-commit draw to choose a random env subset each epoch, plus sketched reference vectors. CountSketch preserved norms ([M] RC-det); dot accuracy through sketches is untested.

## 9. Rollout plan

| stage | blend b | entry gate (all must hold) | exit / kill |
|---|---|---|---|
| **0 Shadow** (live now with the REL v7c rule) | 0.0 | done | — |
| **0b Shadow v9** | 0.0 | v9 pipeline implemented (3-group format, public post-commit reference draw from full envs, leave-task-out, first-solver credit and ledger, max-group null, per-miner gate, window); miners told the format; ≥3 epochs of v9-format data | — |
| **1 Small blend** | 0.05 [P] | G1 **calibration**: real-population per-UID flag rate at α=0.01 ≤ 2% for 3 consecutive epochs (measured 0.9–1.4%); placebo ≤ 0.5. G2 **canary power**: the validator plants blind canary UIDs (1 best-of-16 beyond-policy group per 3-group batch); ≥80% gated within W at the live α (offline at 0.01/210: ≈11–12 batches; e66-like epochs need k≥2). G3 **linearization**: audited batches corr ≥ 0.7, slope 0.7–1.3 (measured 0.76–0.93, 0.87–1.12). G4 **adversarial** ([M] RC-adv baseline): replayed and paraphrase-copied canaries earn ≤ honest-level payment (first-solver credit and ledger work); exact-reference selection is impossible (the draw seed is published only after the deadline); a planted variance-lottery canary is paid only as the curation policy (Q11) allows. G5 **ops**: scores installed within 2 sealed epochs in ≥95% of epochs; writer guard never tripped. G6 public post-commit draw from full envs live, and the reference is reproducible by a third party from the published block hash. | any auto-kill reason → b=0 (signed config, next hourly run; REL `ROLLBACK.md`) |
| **2 Ramp** | 0.10 → 0.20 → … [P] | ≥6 epochs at the previous b with G1–G5 still holding and the drift alarm green; for the first ramp, at least one *real* (non-canary) miner gated, with its paid batches passing the audit | same, plus a drift-alarm trip |
| **3 Steady state** | to decide (Q1) | — | same |

Canaries must be excluded from payment and from the honest null, and listed only in the signed log [P].

## 10. Open questions for Allan

1. **End-state blend.** Should measured impact eventually replace the volume rule, or always ride on top of it? If no one is gated, does the impact share go back to the current rule (the current design) or get burned?
2. **Multiple testing.** Bonferroni over ~210 UIDs (α≈5e-5) needs ≈11–12 batches for 80% k=1 power ([M] RC-adv). BH at q=0.05 needs 5–6, but its FDP exceeded q on e62 (0.13–0.16). Should payment use BH with a conservative pooled null?
3. **Optimizer mismatch.** The score uses plain SGD, but the trainer uses Adam with KL and clipping. Should we score dot(g_b, P·G) with the trainer's second-moment preconditioner P? It is cheap, but untested.
4. **Coverage of J.** J only counts tasks with 20–80% base pass. Should we add a pass@k-style or best-of-N term for hard tasks, or rely on measured transfer (which worked for beyond-policy groups)?
5. **Collusion and paraphrases.** Exact-hash dedup is not enough ([M]: paraphrases pay fully). Is task-level first-solver credit plus a paid-task ledger acceptable? It avoids text-similarity policing, but it means the second solver of a task is paid only for the excess.
6. **Training mix.** Should we train on hits plus random fill (with a cap), all verified data in more, smaller steps, or keep random-256 for cleaner measurement?
7. **The null.** Should it be validator-generated honest groups or the out-of-sample real population? And how should it pool across thousands of envs?
8. **Batch cap and window.** Is 3 batches × 3 groups per epoch with W=4 (12 batches) right, given the measured ≈11–12 batches for 80% power at α=0.01/210?
9. **Gradient subspace.** Is the last-4-layer MLP subspace (604M) representative? Should we check against full-model gradients once, before blend > 0?
10. **Teacher models.** ([M] RC-adv: distilled solutions from 066dfb10 pay like BP and raise held-out J by +4.6.) Should solutions written by a stronger external model be paid like any other improvement? That is consistent with the bitter lesson, but it risks distilling a competitor's model into ours.
11. **Curation vs. data.** Choosing the longest or highest-gradient groups out of many honest samples produces real held-out gains (+5.8) and is paid like new solutions. Should v9 pay miners for selection effort, or should the validator itself train on the top-scoring groups from the ~86% of verified data it currently never trains on, and pay only for gains above that? [P]

## Appendix A. Knobs (complete list)

`b` (blend) · `W` (window epochs, proposed 4 → 12 batches) · `α` (family-wise, proposed 0.01, Bonferroni over N_UID) · batch size (3 groups) · per-UID cap (3 batches/epoch) · audit size (6 batches/epoch). Everything else comes from that epoch's data: reference task filter, q, null quantiles and m_E.

## Appendix B. What is live today vs v9

| item | REL v7c (live, blend 0.0) | v9 |
|---|---|---|
| unit scored | the whole UID commitment (≤9 groups) | each 3-group batch, up to 3 per epoch |
| statistic | raw all-subjects mean, z vs label-shuffle null | max-group p vs honest-group null |
| per-miner | EMA over 6 epochs of score where z>2 | Stouffer + binomial gate, Bonferroni, window W |
| reference seed | derived from the epoch number (public and predictable: exploitable) | public post-commit random draw from full envs (block hash after the deadline) |
| pay | score > 0 | estimated ΔJ above honest median, gated miners only |
| gate result | failed (cross-epoch Spearman ≈0) | new gates G1–G6 (§9). Spearman is the wrong test when honest miners are identical |
