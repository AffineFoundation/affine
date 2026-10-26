# Claimed-env per-batch credit estimator: experiment report

Run on 2026-10-19 on `affine-sampler-8xb200` (8×B200), 14:11–15:27 CR. Aggregates and UIDs only. Code:
`/root/trusted-sampler/claims/` (mirrored on Arbos under `tools/trusted-sampler/claims/`). Results:
`claims/data/claims_results.json`, `claims/data/e62|e64/{gen,dots,fd}-*.json`.

## Verdict
**No: the claimed-env estimator does not separate miners.**
- **Equal budget (24 reference tasks).** Claimed-env and whole-pool scores are both essentially unreliable
  (split-half r −0.02 / +0.19 claimed, −0.23 / −0.08 pool). Only 1–2% of batch pairs are separable.
- **Claims carry no consistent env-specific signal.** In the wrong-claim test, the claimed subjects score higher
  than untouched subjects in one epoch (sign test z +3.2) and *lower* in the other (z −2.2).
- **The claim is gameable.** Naming the "hottest" subject instead of the true claim raises the score on average
  (by +0.2 and +0.5 null SDs; 46% and 56% of batches gain).
- **The whole-reference version works partly.** With ~150 tasks per half (all subjects), there is real, partly
  reliable per-batch signal: split-half r 0.47 / 0.34, Spearman-Brown 0.64 / 0.51, and 15–16% of batches beat the
  label-shuffle null at z>2 (2.3% expected).
- **But it is not a miner trait.** A UID's score in one epoch does not predict its score in the other (Spearman
  0.02, 179 UIDs). Only 3–5% of batch pairs are separable even then.
- **Bottom line.** These scores measure batch-level luck and task-mix effects, not stable miner quality. They
  cannot support per-miner pay without far larger references *and* evidence of persistence.

## What was actually done (method choices)
| step | choice | why |
|---|---|---|
| epochs / bases | training epoch 62 (base `c8a6d342`, step 62→63) and epoch 64 (base `cf791667`, step 64→65) | the two bases named in the plan; 1,727 and 1,718 admitted groups, 205 and 215 UIDs (each UID = one commitment of ≤9 groups = one "batch") |
| batch data | *regenerated*: each admitted group (UID, task index) re-sampled from its base with the network harness (T 1, top-p 0.95, top-k 20, thinking, 8,192 max), 8 rollouts, group contract enforced (≥1 correct, ≥1 non-correct) with ≤2 re-draws; seeds per (uid, task, attempt, j) | real rollout tokens are not readable: the trainer's presigned GETs expired (epoch 64: 12:11 CR), the trainer pod holds no copies, and the R2 RW key is forbidden. Task choice per miner is real (from `AUDIT-POPULATION` records); rollouts are honest base-model samples, so any separation can only come from task choice plus sampling luck |
| batches used | UIDs with ≥5 groups: 193 per epoch; 1,512 / 1,379 valid groups | |
| "envs" | the 7 MATH subjects (Algebra, Intermediate Algebra, Prealgebra, Geometry, Number Theory, Counting & Probability, Precalculus) | SN120 has one env (`affine_math`); subjects are its natural sub-envs |
| claims | synthetic: top subjects covering ≥80% of the batch's groups (typically 2–4 subjects) | as specified |
| reference set | per epoch, 1,400 pool tasks stratified by subject (200 each), 8 base samples each with shared seeds, graded; keep tasks with 2–6/8 correct (20–80%: 330–350 per epoch); split into two disjoint halves; chunks of 4 tasks (≤6 per subject per half, plus 6 pool chunks per half) | fixed, reused for every batch |
| score | first-order (gradient-dot): for an SGD step along the batch's GRPO policy gradient (A = r − group mean incl. live overlong penalty; seq-mean-token-sum-norm, /8192, as in the live trainer), the change in J = mean_tasks[mean logp/len (correct) − mean logp/len (wrong)] (full softmax). Restricted to a 604M-parameter subspace (MLP weights of the last 4 of 32 layers) | one backward per rollout + dot with all reference chunk gradients, so every estimator, placebo and permutation is free. KL term, clipping and Adam preconditioning are omitted |
| equal budget | claimed = 6 chunks (24 tasks), allocated over the claimed subjects in proportion to the claim; pool = 6 random-subject chunks (24 tasks); same half | as specified |
| scratch-step version | real update on a throwaway fp32 copy of the subspace (max per-param update 1e-3), J measured by forward pass on the *held-out* half; for top-2 and bottom-2 batches per epoch | a per-batch real step for all 386 batches would need ~6 min of reference forwards per batch (~40 GPU-h); first-order is the small-step limit |

## Results (per-batch score, units: ΔJ per unit SGD lr)
| epoch | estimator (reference budget) | mean | SD across batches | median label-null SD | split-half r (ρ) | Spearman-Brown | z>2 / z<−2 (of 193) | separable pairs |
|---|---|---|---|---|---|---|---|---|
| 62 | **claimed** (24 tasks) | +1.6e-5 | 5.7e-5 | 4.1e-5 | −0.02 (−0.09) | −0.04 | 10 / 6 | 1.2% |
| 62 | **whole pool** (24 tasks) | +4.9e-6 | 3.2e-5 | 2.5e-5 | −0.23 (−0.24) | <0 | 10 / 2 | 0.3% |
| 62 | wrong claim (24, untouched subjects) | +1.6e-5 | 5.0e-5 | 2.7e-5 | +0.35 (+0.15) | 0.52 | 17 / 2 | 4.6% |
| 62 | all subjects (~150 tasks) | +7.7e-6 | 2.4e-5 | 1.5e-5 | +0.47 (+0.38) | 0.64 | 31 / 7 | 5.2% |
| 64 | **claimed** (24 tasks) | −1.4e-5 | 9.5e-5 | 6.4e-5 | +0.19 (+0.38) | 0.32 | 8 / 12 | 1.9% |
| 64 | **whole pool** (24 tasks) | +1.1e-5 | 3.6e-5 | 2.9e-5 | −0.08 (−0.07) | <0 | 9 / 1 | 0.2% |
| 64 | wrong claim (24) | +1.1e-5 | 4.2e-5 | 2.2e-5 | −0.27 (+0.05) | <0 | 12 / 3 | 0.7% |
| 64 | all subjects (~150 tasks) | +1.3e-5 | 2.7e-5 | 1.3e-5 | +0.34 (+0.21) | 0.51 | 29 / 4 | 3.4% |

**Per-batch SNR.** |z| is measured against the label-shuffle null. Median |z| is 0.83 / 0.86 for claimed vs 0.62 /
0.80 for pool: a slight edge for claimed, but neither is near 2.

**Controls.**
| control | epoch 62 | epoch 64 | reading |
|---|---|---|---|
| zero-step identity | 0 exactly | 0 exactly | pipeline sane |
| shuffled labels (400 within-group permutations per batch) | null mean / null SD = 0.03 | 0.04 | null centred at 0 |
| random-noise advantages (100 per batch) | SD 0.93 × shuffled-null SD | 0.93 × | same noise floor |
| wrong claim (score on untouched subjects − claimed) | median −1.1e-5; claimed higher in 118/192 (sign z +3.2) | median +2.4e-5; claimed higher in 81/193 (z −2.2) | **no consistent env specificity** |
| "name the hottest env" (subject picked on the other half) | Geometry: +1.1e-5 vs true claim (0.21 null SD), 46% of batches gain | Precalc / IntAlg: +3.1e-5 (0.48 null SD), 56% gain | **claiming beats honesty on average: gameable** |
| claimed vs pool rank agreement (Spearman) | 0.11 | 0.29 | different noise, weak shared signal |

**Miner-level checks.**
- Score persistence for the same UID across the two epochs (179 UIDs): all subjects Spearman 0.02, claimed −0.01,
  pool 0.14.
- The batch's subject shares explain R² 0.03 / 0.22 of its all-subjects score, so the task mix is part of what is
  being measured.
- Correlation of the score with the number of valid groups: 0.10 / 0.03.

**Pool vs held-out drift.**
- Only 37 / 46 of 193 batches share any task with the reference set (mean 0.24 / 0.30 groups per batch). Overlap
  vs score Spearman is 0.05 / 0.10, so there is no leakage effect.
- The reference is built from, and evaluated at, the *base* checkpoint of that epoch. Its informative-task set
  (pass 20–80%) and its correct/wrong labels are properties of that policy, so they drift as the policy moves.
- The two epochs used independent reference draws, so drift on fixed tasks was not measured here. The informative
  counts per subject do shift between the bases (Algebra 25→32, Intermediate Algebra 78→61 of 200 drawn).
- A reference must be rebuilt every epoch. Credit computed against a stale pool would reward fixing the old
  policy's weak spots.
- Gradient-dot, all subjects: the top/bottom ranking on half 0 partly holds on half 1. Epoch 62 top-3 half-1 scores are
  +7.6e-5, +2.7e-5, −1.6e-5; bottom-3 are −8.3e-5, −2.8e-5, −1.2e-4. Epoch 64 top-3 half-1 scores are +3.4e-5,
  +1.5e-4, +1.1e-4; bottom-3 are −3.0e-5, +4.0e-5, −1.9e-5.

## Scratch-step check (real update, held-out half)
Top-2 and bottom-2 batches per epoch by the half-0 all-subjects score. Each got a real update on a throwaway fp32
copy of the subspace, θ ← θ + η·g_b with max per-parameter change 1e-3. J was then re-measured by forward pass on
the **held-out half-1** chunks (~42 chunks, ~1,350 sequences).

| epoch | batch (by half-0 rank) | real ΔJ (held-out) | first-order prediction | per-chunk corr(real, pred) | slope |
|---|---|---|---|---|---|
| 62 | top 1 (uid 3) | +1.46e-4 | +1.09e-4 | 0.82 | 0.99 |
| 62 | top 2 (uid 338) | +1.06e-4 | +4.1e-5 | 0.82 | 1.01 |
| 62 | bottom 2 (uid 375) | +4.8e-5 | −1.7e-5 | 0.82 | 0.94 |
| 62 | bottom 1 (uid 130) | −1.00e-4 | −1.15e-4 | 0.92 | 1.12 |
| 64 | top 1 (uid 112) | +1.00e-4 | +2.1e-5 | 0.93 | 1.07 |
| 64 | top 2 (uid 90) | +3.9e-5 | +3.1e-5 | 0.84 | 0.96 |
| 64 | bottom 2 (uid 349) | +2.5e-5 | +3.2e-5 | 0.76 | 0.87 |
| 64 | bottom 1 (uid 261) | −1.05e-4 | −9.1e-5 | 0.92 | 0.96 |

- **Linearization holds.** Real and first-order changes agree chunk-by-chunk (corr 0.76–0.93, slope 0.87–1.12).
  The real step adds a common positive second-order offset of about +1e-5 to +8e-5.
- **Extremes hold out of sample.** Both tops beat both bottoms on held-out tasks in both epochs (8/8 top-vs-bottom
  comparisons).
- **That is the whole-reference (~150 tasks per half) signal, not the claimed-env one.** It is also batch-specific:
  the same UIDs do not persist across epochs.
- **Cost.** ~5 min of reference forwards per evaluation on one B200, i.e. ~10 GPU-min per batch, vs ~1 GPU-s per
  batch for the gradient-dot version.

## Ground truth by real training + generation: not run (infeasible, and why)
- The full 256-group training step moved pass rate by +0.002 ± 0.005 (sampler backfill, ~900 paired tasks). One
  batch is ≤9 groups (≈3.5% of a step), so its true pass-rate effect is ~1e-4.
- At the measured paired SE (≈0.0128 per 128 tasks), resolving 1e-4 needs ~10⁷ paired tasks per batch.
- Training with an inflated lr would make the effect visible, but would no longer measure the real regime.
- The held-out scratch-step check above is the feasible stand-in for ground truth: real update, out-of-sample
  reference.

## Compute and cost
- **This experiment:** generation 8 GPUs × ~17 min (2.3 GPU-h); gradient-dot scoring 7–8 GPUs × ~6.5 min per
  epoch (≈1.7 GPU-h incl. waits); scratch-step check 8 GPUs × 16 min (2.1 GPU-h); HF kernel tests
  ≈0.2 GPU-h; aborted runs ≈0.5 GPU-h. **Total ≈ 6.8 B200 GPU-h.**
- **Wall clock:** about 76 min of the full pod (14:11–15:27 CR), so ≈ $90 of pod time at $71.12/h.
- **Pod:** `affine-sampler-8xb200` (calm-hawk-34), $71.12/h, running since 10:53 CR. ≈ 4.6 h ≈ **$328**
  as of 15:30 CR.
- **Live sampler:** paused 14:11 CR for this experiment, relaunched 15:27 CR:
  `./launch.sh 8 --watch --chain --engine swap -K 256 -m 8`.

## Caveats
- Rollouts are regenerated honest samples, not the miners' actual rollouts. Real batches could differ (e.g.
  curated or adversarial rollouts). Task choice is real.
- The scores are first-order, in a 4-layer MLP subspace, with plain SGD (no Adam preconditioner, no KL). The
  scratch-step check tests linearity at a realistic per-parameter update size.
- Only 2 epochs. Per-subject informative-task counts limit chunks (Algebra and Prealgebra get 3–4 chunks per half,
  not 6). Where a claimed subject ran short, its budget was filled from the other claimed subjects.
