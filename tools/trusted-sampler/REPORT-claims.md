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

## Real rollouts (miners' actual documents), added 2026-10-19 17:00 CR

**Verdict: no; real rollouts don't change the picture.** The extra data pins down the same conclusion.
- **Within an epoch:** scoring against the whole reference (all subjects) shows only weak split-half reliability.
- **Across epochs:** UID scores don't persist (Spearman between −0.12 and +0.06 for every epoch pair).
- **Separability:** about 1% of batch pairs can be told apart.
- **Labels:** our verifier finds no wrong miner labels, so there is no label-quality signal to reward or punish.
- **Staleness:** every batch was built on that epoch's base checkpoint, so staleness can't explain or test anything.
- **Bottom line:** the estimator can't serve as per-miner credit.

**Data.** Read-only GETs from the lifecycle bucket (`private/affine-mimo9b-grpo-20261013-e0000NN/training/<miner>/<slot>.json`) for
training steps 62–66. For steps 62–65 that is every admitted group: 1727 / 1797 / 1719 / 1831, with 256 trained per step. Each document
carries the miner's prompt tokens, completion tokens, claimed classification and reward, and the checkpoint the batch was built on.
- **Step 66** (1881 docs, not yet trained) was fetched but not scored, to meet the deadline.
- **Prompts:** in 100% of documents the miner's prompt tokens are identical to our chat-template prompt.
- **Pipeline:** unchanged (`grad_real.py`): same subspace, chunks and estimator.
- **References:** the existing references were reused for steps 62 and 64. New references were built from each base for steps 63 and 65.
- **Advantages:** A = r − group mean, with r = overlong_reward(claimed correct, length).

| train step | batches | claimed labels wrong (our verifier) | all-subj split-half r (ρ) | Spearman-Brown | z>2 / z<−2 | separable pairs | claimed SH r (ρ) | pool SH r (ρ) | placebo: shuffled mean/SD | noise SD / shuffled SD |
|---|---|---|---|---|---|---|---|---|---|---|
| 62 | 205 | 0 / 13,816 | +0.05 (+0.38) | 0.09 | 38 / 6 | 0.5% | +0.22 (+0.07) | −0.53 (−0.17) | 0.037 | 0.93 |
| 63 | 211 | 0 / 14,376 | +0.19 (+0.36) | 0.32 | 39 / 2 | 1.0% | −0.15 (+0.15) | +0.69 (+0.55) | 0.036 | 0.93 |
| 64 | 215 | 0 / 13,752 | +0.20 (+0.13) | 0.33 | 28 / 0 | 1.7% | −0.29 (+0.16) | −0.39 (−0.17) | 0.035 | 0.92 |
| 65 | 218 | 0 / 14,648 | +0.53 (−0.13) | 0.69 | 16 / 3 | 0.9% | −0.28 (−0.29) | −0.88 (−0.50) | 0.037 | 0.93 |

**Cross-epoch UID stability (all-subjects score, Spearman over shared UIDs, about 200 per pair):**

| pair | 62–63 | 62–64 | 62–65 | 63–64 | 63–65 | 64–65 |
|---|---|---|---|---|---|---|
| all-subjects | +0.04 | −0.08 | −0.08 | +0.02 | −0.12 | +0.06 |
| claimed | +0.18 | +0.06 | +0.14 | +0.12 | −0.01 | +0.08 |
| pool | +0.08 | +0.15 | +0.12 | +0.08 | +0.01 | +0.07 |

- **Averaging epochs doesn't help.** Per-epoch z-scored UID means over steps {62, 64} vs {63, 65} give Spearman +0.01 (Pearson +0.21, carried by
  outliers) across 198 UIDs.
- **Heavy tails.** Within an epoch, Pearson and Spearman split-half often disagree in sign (e.g. step 65: +0.53 vs −0.13). A few
  extreme batches dominate the score; there is no stable ranking.
- **What z>2 means here.** The count is z against the shuffled-label null, and it is mostly positive (16–39 batches per epoch vs 0–6 negative).
  That is the common effect of honest labels: training on them raises correct-minus-wrong log-prob. It shows that labels are honest, not that
  miners differ.

**Label error rate.** We re-graded all 56,592 real rollouts (steps 62–65; step 66 also 0 of 15,048) with our local verifier replica.
- **Agreement:** 100% with the miners' classification, with 0 indeterminate.
- **Score vs label error:** undefined, because the error rate is 0 everywhere.
- **Sensitivity check (synthetic flips):** flipping a random 10 / 25 / 50 / 100% of a batch's labels moves its score by only −0.1…−0.2 /
  −0.5 / −0.9 / −0.8…−2.0 null SDs. Even fully inverted labels move a batch by about 2 SD at most, so this score is also a weak label-fraud
  detector per batch.

**Staleness.** 100% of documents in every epoch are built on that epoch's base checkpoint. Admission enforces this, so a score-vs-staleness
relation can't be measured (no variance).

**Controls.**
- **Placebo:** as before (shuffled-label null mean ≈ 0.036 SD, noise/shuffled SD 0.93, zero step = 0).
- **Wrong claim:** a batch's own claimed subjects score higher than unclaimed ones in 127–147 of 204–218 batches (sign z +2.7 to +5.8). With real data the
  specificity is consistent, but small.
- **Hottest-subject claim:** naming the hottest subject instead of the real claim changes the score by −0.5 to +0.02 null SD. With real data
  the attack doesn't pay, but the honest claim's own advantage is also too small to separate miners.
- **Other checks:** fraction of trained groups vs score ρ = −0.07…+0.06; overlap with reference tasks vs score ρ ≈ 0.

**Regenerated vs real** (same UIDs, same reference; steps 62 / 64):
- **Score correlation:** all-subjects score Spearman −0.09 / +0.13.
- **Spread:** across-batch SDs are similar (2.1e-5 real vs 2.4e-5 / 2.7e-5 regenerated).
- **Conclusion:** the regenerated-batch scores were noise at the same level as the real ones, and the earlier verdict didn't depend on the
  regeneration.

**Not done (deadline):**
- the step 66 score;
- scratch-step checks on the real top/bottom batches.

**Compute.**
- **This rerun:** about 4.7 B200 GPU-h (GPUs 4–7, 15:48–16:58 CR). Live-sampler instances on GPUs 4–7 were paused for it; GPUs 0–3 kept
  running the live sampler throughout.
- **Total for the claims experiment:** about 11.5 GPU-h.

**Files (pod):**
- `claims/grad_real.py`, `regrade_real.py`, `analyze_real.py`, `summary_real.py`, `fd_real.py` (not run);
- `claims/data/real/real-6{2..6}.jsonl.gz` (miner docs; private);
- `claims/data/real/regrade-*.json`, `claims/data/e6{2..5}/dots-real-*.json`;
- `claims/data/claims_real_results.json`.

**Credential.** A new read-only R2 token ("Workers R2 Storage Bucket Item Read", scoped to `affine-verification-mock-20260930`) was created
and stored next to the lifecycle's credential at Arbos `state/private/r2-readonly.env` (0600, same `R2_ACCESS_KEY_ID`/`R2_SECRET_ACCESS_KEY`
format). Its list and get reads were verified. It is not in 1Password.

## Planted batches (known-quality synthetic miners), added 2026-10-19 18:45 CR

**Question.** Real miners appear to run the same honest code. Does the all-subjects gradient-dot score separate batches whose quality
really differs?

**Answer: only for large differences.**
- **Beyond-policy batches are detected easily.** These carry correct solutions to tasks the base solves ≤1/8 of the time. They score 7–8 SD
  above the real-miner population, with AUC 1.00 vs honest in both epochs and every planted batch distinguishable.
- **Fully inverted labels are detected weakly.** AUC 0.06 / 0.19 that they score below honest; 3 of 10 and 0 of 10 batches are
  distinguishable from honest.
- **Everything else is inside the honest spread:** random rewards, 30% inverted labels, garbage marked correct, easy-only, hard-only, stale
  batches, and "good" balanced 20–80% groups from the current model.
- **So:** the score does see real quality differences, but only ones far larger than anything an honest-code miner population produces.
  This is consistent with the hypothesis that the real population simply has no variance to detect.

**Setup**
- **Bases:** the brief's "steps 63 / 65" are train steps 62 / 64 here (bases c8a6d342 / cf791667). Each planted batch has 9 groups, the
  median real batch size.
- **Count:** 10 batches per type per epoch, except good-balanced (9 at step 64) and beyond-policy (9 / 8), which were limited by the
  available pools.
- **Scoring:** the identical pipeline, i.e. the same subspace, the same reference V (84 chunks, both halves), and A = r − group mean with
  the overlong penalty.
- **Comparison set:** scores are compared against the 205 / 215 real miner batches of the same epoch. z_null uses 400 within-group
  label shuffles per batch.

**How each type was made**
1. **Honest:** randomly resampled real miner groups.
2. **Random rewards:** real groups with Bernoulli(0.5) labels.
3. **Faked labels:** real groups with labels inverted, 100% (3a) or a random 30% (3b).
4. **Garbage marked correct:** in real groups, every correct completion was replaced by a truncated (first 20%) or token-shuffled version
   and kept at reward 1; wrong completions were kept.
5. **Easy-only:** base reference samples on tasks with ≥7/8 base pass.
6. **Hard-only:** base reference samples on tasks with ≤1/8 base pass. These are never part of reference chunks.
7. **Good balanced:** base-model groups with 3–5 of 8 correct, on miner-chosen tasks that are not reference tasks.
8. **Beyond-policy:** for 160 reference tasks with ≤1/8 base pass, 40 extra base samples each (best-of-48 in total, not 64, for time).
   Correct ones were found on 81 / 78 tasks; each group is up to 2 found correct plus wrong base samples. The solutions come from the base
   model itself via search, not from a stronger model.
9. **Stale:** real honest groups from 2 steps earlier (step 60 docs on 2198992b, and step 62 docs on c8a6d342), scored at the current base.

| type | step 62: mean score | z_null | frac z>2 / z<−2 | z vs real pop | # distinguishable (|z_pop|>2) | step 64: mean score | z_null | frac z>2 / z<−2 | z vs real pop | # distinguishable |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 beyond-policy (bo48) | +1.96e-4 | +2.30 | 0.7 / 0 | **+6.9** | **9/9** | +3.29e-4 | +3.31 | 1.0 / 0 | **+8.4** | **8/8** |
| 7 good balanced | +1.43e-5 | +0.50 | 0.1 / 0 | −0.10 | 2/10 | +3.93e-5 | +1.10 | 0.2 / 0 | +0.69 | 2/9 |
| 6 hard-only | +2.92e-5 | +0.89 | 0.2 / 0 | +0.47 | 1/10 | +1.33e-5 | +0.93 | 0.2 / 0 | 0.00 | 1/10 |
| 5 easy-only | +1.64e-5 | +0.77 | 0.1 / 0 | −0.02 | 0/10 | +2.83e-5 | +1.26 | 0.3 / 0 | +0.40 | 0/10 |
| 1 honest (control) | +7.4e-6 | +0.40 | 0 / 0 | −0.37 | 0/10 | +1.71e-5 | +1.28 | 0.2 / 0 | +0.10 | 0/10 |
| 9 stale (2 steps) | +2.8e-6 | +0.23 | 0 / 0 | −0.54 | 1/10 | +7.7e-6 | +0.52 | 0 / 0 | −0.15 | 0/10 |
| 4 garbage marked correct | +2.21e-5 | +0.59 | 0 / 0 | +0.20 | 0/10 | −1.35e-5 | −0.86 | 0 / 0.3 | −0.71 | 0/10 |
| 3b labels 30% inverted | −5.2e-6 | −0.26 | 0 / 0 | −0.85 | 1/10 | −3.6e-6 | −0.18 | 0 / 0 | −0.45 | 0/10 |
| 2 random rewards | −7.8e-6 | −0.36 | 0 / 0.1 | −0.95 | 1/10 | −5.0e-6 | −0.23 | 0.1 / 0.1 | −0.48 | 0/10 |
| 3a labels 100% inverted | −2.37e-5 | −1.55 | 0 / 0.3 | −1.56 | 3/10 | −3.7e-6 | −0.18 | 0 / 0 | −0.45 | 0/10 |

Reference spread: real miner batches have SD 2.6e-5 / 3.8e-5, and 18.5% / 13.0% of them have z_null > 2.

**AUC (score separating the groups; honest = planted honest + all real miner batches):**
- **good (7+8) vs honest:** 0.72 / 0.81. Beyond-policy alone: 1.00 / 1.00. Good-balanced alone: 0.47 / 0.65.
- **honest vs bad (2, 3a, 3b, 4):** 0.74 / 0.78. Per type: 3a 0.94 / 0.81, 3b 0.77 / 0.74, random 0.78 / 0.77, garbage 0.46 / 0.78.
- **good vs bad:** 0.83 / 0.92.

**Ranking and cross-epoch consistency**
- **Step 62 ranking:** 8 > 6 > 4 > 5 > 7 > 1 > 9 > 3b > 2 > 3a.
- **Step 64 ranking:** 8 > 7 > 5 > 1 > 6 > 9 > 3b > 3a > 2 > 4.
- **Consistency:** type means correlate Spearman +0.53 / Pearson +0.97 across the two epochs. The Pearson is carried by beyond-policy at the
  top. Below it the order is mostly noise, apart from bad types sitting slightly below honest in both epochs.
- **Within-type spread:** typically 1–4e-5 per batch, about the size of the whole real-population SD. Two draws of the same type (built
  before and after a rebuild) gave good-balanced means of 2.6e-5 vs 1.4e-5.

**Scratch-step check (real update on a throwaway fp32 copy, max per-parameter change 1e-3, J on held-out half 1)**
- **Labels, good and hard batches:** the first-order prediction tracks the real update chunk by chunk.
  - Step 62: faked-100% corr 0.86 (slope 1.10), good corr 0.92 (1.11), hard-only corr 0.91 (1.14).
  - Step 64: faked-100% corr 0.91 (1.01), good corr 0.90 (0.98), hard-only corr 0.76 (0.72).
- **Garbage is the exception** (corr 0.24). Its gradient is dominated by huge per-token gradients on nonsense text, so η is tiny (0.13) and
  the real change is mostly the common second-order offset.
- **Sign:** the step-64 faked batch moves J down in both prediction and reality (−8.6e-5 / −6.9e-5). The step-62 faked batch-0 is one of the
  faked batches that happens to score positive, again in both.
- **Caveat:** the step-62 good-0 and hard-0 checks used the first draw of those types, before the rebuild that added type 8.

### Scoring rules (same gradients, no new GPU runs)

| rule | AUC good vs honest (62/64) | AUC beyond-policy vs honest | AUC good-balanced vs honest | AUC honest vs bad | type-rank ρ 62↔64 | real-miner cross-epoch UID ρ, mean [min, max] over 6 pairs (steps 62–65) |
|---|---|---|---|---|---|---|
| (a) raw grad-dot (current; no clipping is applied anywhere in this pipeline) | 0.72/0.81 | 1.00/1.00 | 0.47/0.65 | 0.74/0.78 | +0.53 | −0.03 [−0.12, +0.06] |
| (b1) z vs shuffled-label null (size-normalized) | 0.59/0.75 | 0.81/0.97 | 0.40/0.56 | 0.79/0.79 | +0.59 | −0.02 [−0.12, +0.09] |
| (b2) cosine in reference-chunk space (mean/rms over chunks) | 0.56/0.66 | 0.69/0.90 | 0.44/0.44 | 0.81/0.77 | +0.76 | −0.01 [−0.13, +0.09] |
| (c1) per-subject, mean over subjects | 0.71/0.75 | 0.94/1.00 | 0.51/0.54 | 0.66/0.75 | +0.70 | −0.02 [−0.13, +0.12] |
| (c2) per-subject, max over subjects | 0.86/0.87 | 0.98/0.99 | 0.75/0.76 | 0.51/0.51 | +0.66 | +0.11 [−0.01, +0.30] |
| (d) one reference draw (half 0) vs (a) = two draws averaged | 0.72/0.75 | 0.99/1.00 | 0.47/0.54 | 0.71/0.65 | +0.73 | +0.01 [−0.16, +0.07] |
| (e) loss-agnostic SFT gradient of claimed-correct completions | 0.38/0.15 | 0.51/0.01 | 0.26/0.27 | 0.77/0.95 | +0.65 | +0.19 [+0.12, +0.29] |

- **No rule gives stable credit for real miners.** Cross-epoch UID consistency is nil for (a)–(d). Only (c2) max-subject and (e) SFT show
  small positive persistence.
  - For (e) this is expected: the SFT score is a per-miner trait (subject/length mix of correct completions), not quality. It ranks
    beyond-policy batches at or below honest (AUC 0.51 / 0.01) and rewards easy-only most.
  - (c2) is the best overall trade-off on the planted set: good-balanced AUC 0.75 / 0.76, while staying 0.98 vs beyond-policy. But it loses
    all power against bad batches (honest-vs-bad 0.51), because a max over subjects lets any batch pick its best subject.
- **True gradient-norm cosine wasn't possible.** Per-batch gradient norms were not stored, so (b) uses two proxies; a real cosine needs one
  extra backward pass per batch. (d) has only two independent reference halves per epoch: averaging them improves the good-balanced AUC
  slightly (0.47/0.54 → 0.47/0.65) and the honest-vs-bad AUC (0.71/0.65 → 0.74/0.78).

**Compute (planted + rules)**
- **GPU time:** about 4.0 B200 GPU-h.
  - Beyond-policy sampling 2.15 h (GPUs 3 and 6, about 65 min each).
  - Planted gradient-dots 0.5 h.
  - Scratch-step checks 1.3 h.
  - The scoring rules used CPU only.
- **Live sampler:** paused on GPUs 1–3 for this experiment and relaunched there (16:44 and 18:41 CR); GPUs 0–3 are running it now. GPUs
  4–7 are with the release scorer.
- **Pod cost:** pod total 7.8 h × $71.12 ≈ $555 (as of 18:41 CR).

**Files (pod `claims/`)**
- Scripts: `plant_build.py`, `plant_teacher_gen.py`, `plant_dots.py`, `plant_analyze.py`, `plant_rules.py`, `fd_plant.py`.
- Data: `data/planted/e6{2,4}/` (spec, new sequences, dot caches, teacher samples, scratch-step results).
- Results: `data/planted_results.json`, `data/planted_rules.json`.

## Detection threshold (Oct 19, 2026, 7:02–8:10 PM CR)

**Verdict:** The current batch score (raw) catches whole beyond-policy batches at every best-of-N tested, but it misses a miner that adds 1–2 new solutions to an otherwise honest 9-group batch: detection is 6–15%. A per-group max rule raises that to about 40–55% on steps 62/64 and about 20% on e66. The true gradient-norm cosine rule is worse than raw for this use. No rule × batch-size combination tested reaches 80% power for k=1–2 in a single batch. This needs per-miner aggregation across batches, or a stronger per-group signal.

### Setup
- **Data and base checkpoints:** steps 62 and 64, plus e66. For e66 the base is 23a03720 (the base the e66 miners built on); 066dfb10 is its output. For e66 I reused the release scorer's reference (80 chunks, verified identical) and its real-rollout dots.
- **Real population:** UID batches only (n = 205 / 215 / 218; the 40 e66 docs with uid None are excluded).
- **Beyond-policy groups:** one hard reference task each (≤1/8 correct), with the first ≤2 correct teacher samples among the first N draws plus 6 wrong reference samples. Teacher draws were 40 on e62/e64 and 16 on e66, so e66 has no bo40 level.
- **Group pools** (bo4 / bo8 / bo16 / bo40):
  - e62: 23 / 43 / 60 / 81;
  - e64: 28 / 41 / 58 / 78;
  - e66: 39 / 64 / 80 / –.
- **Pure-dose batches:** 10 per level, 9 groups each; groups are resampled when the pool has fewer than 90.
- **Mixtures:** 400 draws per k, with k beyond-policy groups replacing honest real groups.
- **"Detected"** means z > 2 against the real population. For maxgroup, the null is the distribution of honest batches' maximum group z (false-positive rate 4–7%).
- **Cosine rule:** true gradient-norm cosine = raw / ‖g_b‖. ‖g_b‖ comes from per-rollout CountSketches (D = 16384, so any batch or mixture norm can be computed; ‖sk‖²/‖g‖² = 0.999 ± 0.011 over 17.9k rollouts). Sketches stored at `claims/data/gnorm/e{62,64,66}/`.

### Dose response (pure beyond-policy batches): mean z_pop / detection rate / AUC vs real

| level | e62 raw | e62 maxgroup | e62 cos | e64 raw | e64 maxgroup | e64 cos | e66 raw | e66 maxgroup | e66 cos |
|---|---|---|---|---|---|---|---|---|---|
| bo4 | 8.0 / 1.0 / 1.00 | 7.5 / 1.0 / 1.00 | 1.4 / 0.1 / 0.88 | 7.7 / 1.0 / 1.00 | 3.6 / 1.0 / 0.98 | 2.7 / 0.8 / 0.99 | 3.7 / 0.8 / 0.90 | 3.5 / 0.9 / 0.96 | 0.9 / 0.1 / 0.75 |
| bo8 | 5.1 / 0.9 / 0.92 | 5.6 / 1.0 / 0.99 | 0.6 / 0.0 / 0.72 | 8.4 / 1.0 / 1.00 | 4.9 / 1.0 / 0.99 | 2.8 / 1.0 / 1.00 | 3.6 / 0.8 / 0.89 | 3.4 / 0.9 / 0.96 | 0.8 / 0.1 / 0.74 |
| bo16 | 5.3 / 0.7 / 0.83 | 6.3 / 0.9 / 0.99 | 0.5 / 0.1 / 0.64 | 7.6 / 1.0 / 1.00 | 4.8 / 1.0 / 0.99 | 2.3 / 0.6 / 0.98 | 4.5 / 0.9 / 0.97 | 3.4 / 0.8 / 0.96 | 1.1 / 0.2 / 0.77 |
| bo40 (≈ the earlier bo48) | 7.2 / 1.0 / 1.00 | 7.2 / 1.0 / 1.00 | 0.9 / 0.0 / 0.79 | 7.9 / 1.0 / 0.99 | 4.8 / 1.0 / 0.99 | 2.4 / 0.7 / 0.97 | – | – | – |

Signal does not rise with N. Every beyond-policy group carries one or two correct solutions on a task the policy almost never solves, so per-group strength is roughly the same at every N. Larger N only adds tasks to the pool. **The dose that matters is how many such groups are in the batch.**

### Mixtures (9-group batch, k beyond-policy groups): detection rate for raw / maxgroup / cos

| k of 9 | e62 bo4 | e62 bo8 | e62 bo16 | e64 bo4 | e64 bo8 | e64 bo16 | e66 bo4 | e66 bo8 | e66 bo16 |
|---|---|---|---|---|---|---|---|---|---|
| 0 (false-positive rate) | .01/.04/.01 | .01/.06/.01 | .00/.04/.01 | .03/.05/.02 | .01/.06/.03 | .03/.05/.02 | .02/.05/.02 | .03/.07/.01 | .03/.07/.03 |
| 1 | .12/.54/.01 | .07/.41/.00 | .11/.45/.00 | .06/.46/.04 | .11/.42/.08 | .10/.41/.08 | .08/.21/.03 | .10/.21/.04 | .09/.21/.03 |
| 3 | .56/.91/.02 | .40/.81/.01 | .47/.79/.01 | .72/.84/.30 | .61/.83/.26 | .62/.81/.18 | .30/.42/.06 | .33/.48/.05 | .38/.53/.06 |
| 5 | .90/.99/.03 | .64/.92/.00 | .77/.94/.01 | .96/.95/.60 | .93/.95/.44 | .90/.88/.30 | .50/.60/.07 | .60/.66/.10 | .64/.72/.10 |

**Minimal detectable improvement** (smallest k of 9 with ≥80% power at z > 2):

| rule | e62 (bo4/8/16/40) | e64 (bo4/8/16/40) | e66 (bo4/8/16) |
|---|---|---|---|
| raw | 5 / 7 / 6 / 6 | 4 / 4 / 5 / 4 | >9 / 9 / 8 |
| maxgroup | 2 / 3 / 4 / 3 | 3 / 3 / 3 / 3 | 9 / 8 / 7 |
| cos | none ≤ 9 | 8 / none / none / none | none |

As a fraction of the batch, maxgroup needs about 1/3 of groups to be beyond-policy on 62/64. On e66, raw and maxgroup need nearly the whole batch.

### Batch size (k = 1 or 2 bo16 groups in a B-group batch vs a resampled honest null): power raw / maxgroup

| B | e62 k=1 | e62 k=2 | e64 k=1 | e64 k=2 | e66 k=1 | e66 k=2 |
|---|---|---|---|---|---|---|
| 3 | .36/.51 | .64/.79 | .50/.54 | .77/.74 | .32/.42 | .51/.58 |
| 6 | .33/.48 | .56/.69 | .43/.51 | .74/.76 | .18/.26 | .39/.47 |
| 9 | .27/.42 | .49/.68 | .27/.35 | .61/.57 | .19/.27 | .36/.40 |
| 18 | .15/.39 | .30/.57 | .19/.29 | .47/.45 | .12/.11 | .23/.20 |
| 36 | .15/.34 | .27/.59 | .14/.14 | .27/.19 | .10/.14 | .20/.17 |

For bo40 groups at B = 3, k = 2 is .65/.78 on e62 and .85/.84 on e64.

**Ceiling:** a single beyond-policy group against a single honest group has an AUC of only 0.77 / 0.89 / 0.74 (e62 / e64 / e66). Halving the reference (one half instead of both) changed power inconsistently: about the same on e62, mixed on e64, slightly lower on e66. The limit is the honest groups' own heavy-tailed spread, not reference noise, so a bigger reference alone will not fix it.

### Rule comparison including the true cosine (all planted + real batches, 62/64)

| rule | AUC good vs honest | AUC beyond-policy vs honest | AUC good-balanced vs honest | AUC honest vs bad | type-rank ρ 62↔64 | real-UID ρ across epochs |
|---|---|---|---|---|---|---|
| raw | 0.72/0.81 | 1.00/1.00 | 0.47/0.65 | 0.74/0.78 | +0.53 | −0.03 [−0.12, +0.06] |
| **cos_true** | 0.60/0.74 | 0.79/0.98 | 0.43/0.52 | 0.83/0.79 | **+0.94** | −0.04 (62↔64 only) |
| z_null | 0.59/0.75 | 0.81/0.97 | 0.40/0.56 | 0.79/0.79 | +0.59 | −0.02 |
| chunk_cos | 0.56/0.66 | 0.69/0.90 | 0.44/0.44 | 0.81/0.77 | +0.76 | −0.01 |
| subj_mean | 0.71/0.75 | 0.94/1.00 | 0.51/0.54 | 0.66/0.75 | +0.70 | −0.02 |
| subj_max | 0.86/0.87 | 0.98/0.99 | 0.75/0.76 | 0.51/0.51 | +0.66 | +0.11 |
| one_draw | 0.72/0.75 | 0.99/1.00 | 0.47/0.54 | 0.71/0.65 | +0.73 | +0.01 |
| sft | 0.38/0.15 | 0.51/0.01 | 0.26/0.27 | 0.77/0.95 | +0.65 | +0.19 |

The true cosine gives the most stable type ranking and good honest-vs-bad separation. It is weak on exactly the target case: new correct solutions are low-probability sequences with large gradients, so dividing by ‖g_b‖ removes most of their advantage. It still does not separate real miners (ρ ≈ 0).

### Recommendation (to catch a miner adding ~1–2 genuinely new solutions per batch)
1. **Scoring rule:** use a per-group test (max over groups of the group z, with a calibrated null from honest groups) as the detector, not raw, cosine, or other batch means. It is about 2–4× more sensitive at k = 1–2 and holds a 4–7% false-positive rate. Keep raw (or the cosine) only as the batch-quality score; it catches whole bad or beyond-policy batches.
2. **Batch size:** use small batches of about 3 groups (24 rollouts). Power drops steadily as B grows; at B = 3, k = 2 reaches 0.74–0.85 on e62/e64. At the current median of 9 groups, k = 1 is 0.35–0.45 and k = 2 is 0.57–0.68.
3. **Aggregate per miner:** single batches cannot reach 80% power for one new solution (e62/e64 ≤ 0.6, e66 ≤ 0.4). A miner that consistently adds 1–2 should be flagged by combining its batches' max-group z across an epoch, e.g. Stouffer's method or a count of z > 2 batches against the 4–7% false-positive rate. Mean max-group z at k = 1 is 1.1–2.7 per 9-group batch, so about 4–9 batches should give roughly z ≥ 4 combined. This is an extrapolation and was not simulated.
4. **Calibrate per epoch:** sensitivity varies a lot by epoch (e66 is about half of e64). Thresholds have to come from each epoch's honest-group null, not a fixed constant.

**Compute this round:**
- GPU time: about 2.6 B200-hours (e66 teacher 0.55, sketches e62/e64 1.25, e66 dots 0.25, e66 sketches 0.55). Analyses ran on CPU.
- Pod time: about 1.1 h this round (≈ $80).
- Artifacts:
  - scripts: `claims/gnorm.py`, `run_gnorm.sh`, `plant_dose.py`, `plant_rules.py` (now includes cos_true);
  - results: `data/dose_results.json` (62/64/66), `data/dose-all.out`, `data/planted_rules.json`.
- Miner data stays private on the pod; only aggregates are reported.

## Per-miner aggregation (Oct 19, 2026, 8:10–8:30 PM CR)

**Verdict:** Combining a miner's batches makes a 1–2-new-solution miner detectable, provided each batch gets a calibrated p-value and the batches are combined with a calibrated test.
- **Method:** each batch's max-group p-value comes from the epoch's honest-group null at that batch's size; the batches are combined with Stouffer's method, or with a binomial count of hits against the per-batch false-positive rate.
- **Power and false positives:** with 3-group batches and 1 beyond-policy group per batch, Stouffer reaches ≥80% power at about 6 batches on e62/e64 and 8–10 on e66. Its per-miner false-positive rate stays at 0–5%; the binomial test stays at 0–3%.
- **Naive aggregation is gameable:** flagging a miner if *any* batch hits, or summing raw z, is miscalibrated. An honest miner who splits 36 groups into 36 one-group batches is flagged 20–61% of the time, against 4–11% for a single 36-group batch.
- **Real miners:** 213 actual UIDs, aggregated over e62–e66 (one batch per epoch, mean 4.9 batches), give the honest rates. Binomial flags 0.9% and Stouffer 1.4% (α = 0.01), while any-hit flags 22% and naive summed z 8.5%. The hit-count histogram (0: 167, 1: 40, 2: 6) matches independent batches at q ≈ 0.06, so no real UID consistently submits beyond-policy groups.

### Method (`claims/miner_agg.py`, CPU only, stored gradient dots)
- **Group score:** the same reference weighting as the pipeline.
- **Calibration and background:** per epoch, honest real groups (≈1.7–1.9k) are split into two halves, CAL (null) and SIM (background), so calibration is out-of-sample.
- **Batch p-value:** P_null(max over S groups ≥ observed), from 200k resampled CAL batches of the same size S.
- **Per-batch threshold q:** the batch counts as a hit when p < q, where q is the false-positive rate of the old "max-group z > 2" threshold at that size. q is 0.03 (S = 3) and 0.06–0.07 (S = 9).
- **Beyond-policy groups:** from the best-of-16 pool (60 / 58 / 80 groups).
- **Simulated miners:** 2,000 per configuration.
- **Per-miner tests at α = 0.01:**
  - any: any batch hits;
  - binomial: P(Binom(B, q) ≥ hits) < α;
  - Stouffer: Σ Φ⁻¹(1 − p_b)/√B > 2.33;
  - naive summed z: Σ z_b/√B > 2.33.

### Per-miner power and false-positive rate: Stouffer / binomial (e62 · e64 · e66)

| batch size, BP groups per batch | B=3 | B=4 | B=6 | B=8 | B=10 | B=12 |
|---|---|---|---|---|---|---|
| 3 groups, k=0 (FP) | .01/.00 · .01/.00 · .02/.01 | .01/.00 · .00/.00 · .02/.00 | .01/.00 · .00/.00 · .03/.00 | .01/.00 · .00/.00 · .02/.00 | .01/.00 · .00/.00 · .02/.01 | .01/.00 · .00/.00 · .02/.00 |
| 3 groups, k=1 | .67/.45 · .73/.46 · .44/.33 | .75/.63 · .84/.65 · .52/.16 | .86/.57 · .95/.61 · .65/.41 | .91/.79 · .98/.82 · .79/.66 | .95/.91 · .99/.91 · .86/.80 | .98/.96 · 1.0/.97 · .91/.76 |
| 3 groups, k=2 | .91/.80 · .97/.81 · .79/.66 | .95/.92 · .99/.92 · .86/.48 | .99/.94 · 1.0/.94 · .94/.82 | .99/.99 · 1.0/.99 · .98/.95 | 1.0 · 1.0 · .99/.98 | 1.0 · 1.0 · 1.0/.98 |
| 9 groups, k=0 (FP) | .02/.00 · .01/.00 · .05/.00 | .02/.00 · .01/.00 · .04/.00 | .01/.00 · .01/.00 · .04/.02 | .01/.00 · .00/.00 · .04/.01 | .01/.00 · .00/.00 · .05/.02 | .01/.00 · .00/.00 · .05/.03 |
| 9 groups, k=1 | .61/.07 · .40/.06 · .26/.03 | .70/.18 · .53/.16 · .36/.10 | .81/.45 · .72/.39 · .48/.29 | .89/.68 · .86/.64 · .63/.23 | .93/.62 · .93/.55 · .70/.39 | .97/.77 · .95/.71 · .76/.54 |
| 9 groups, k=2 | .88/.23 · .81/.23 · .52/.11 | .93/.54 · .92/.49 · .63/.27 | .99/.86 · .99/.83 · .84/.60 | 1.0/.98 · 1.0/.96 · .94/.60 | 1.0/.97 · 1.0/.94 · .97/.80 | 1.0/.99 · 1.0/.98 · .99/.92 |

For comparison, the miscalibrated rules at k=0 (honest miners), 9-group batches: any-hit FP climbs from 0.09/0.06/0.27 at B=3 to 0.31/0.23/0.73 at B=12. Naive summed z gives 0.04–0.29.

**Batches needed for 80% power (Stouffer):**

| case | e62 / e64 | e66 |
|---|---|---|
| 3-group batches, 1 new solution | ≈ 6 | ≈ 10 |
| 9-group batches, 1 new solution | ≈ 8 | > 12 |
| 3-group batches, 2 new solutions | ≈ 3 | ≈ 4–6 |
| 9-group batches, 2 new solutions | ≈ 4 | ≈ 6 |

Binomial needs roughly 1.5–2× as many batches; it is conservative and discrete.

### Gaming check: an honest miner splitting a fixed 36-group budget into 36/S batches (per-miner flag rate: any / binomial / Stouffer)

| S (groups per batch) | 1 | 2 | 3 | 4 | 6 | 9 | 12 | 18 | 36 |
|---|---|---|---|---|---|---|---|---|---|
| e62 | .42/.003/.013 | .32/.006/.011 | .23/.002/.007 | .22/.001/.013 | .16/.000/.013 | .11/.001/.017 | .04/.000/.026 | .04/.000/.039 | .04/.000/.035 |
| e64 | .20/.001/.002 | .18/.000/.001 | .11/.000/.002 | .14/.000/.002 | .09/.000/.003 | .08/.000/.011 | .07/.000/.011 | .10/.002/.035 | .09/.000/.043 |
| e66 | .61/.003/.009 | .56/.006/.015 | .47/.003/.021 | .45/.003/.024 | .37/.006/.032 | .36/.004/.042 | .20/.002/.063 | .21/.000/.080 | .11/.000/.081 |

- **Yes, max-group is gameable** if any flagged batch counts. Many tiny batches multiply the chances (multiple testing): up to 61% flagged with no new solutions. If flags earn reward, honest miners are paid to split.
- **Per-miner correction removes this.** The binomial test over the miner's batch count stays at ≤0.6% for every split. Stouffer stays at 0.1–4% for S ≤ 9.
- **Stouffer creeps up to 4–8% at S ≥ 12.** The far tail of the max over many groups is poorly estimated from ≈900 calibration groups. Larger batches should use a parametric or pooled-epoch tail, or a fixed batch size.
- **Known gap, not simulated:** cherry-picking lucky honest groups would need the private reference gradient.

### Recommendation
1. **Batch format:** a fixed small batch size of 3 groups, so every batch is scored on the same null and splitting gains nothing.
2. **Per-batch statistic:** the max-group p-value against that epoch's honest-group null. Calibrate per epoch, out-of-sample.
3. **Per-miner decision:** use Stouffer's method over all of the miner's batches in the window as the score, and the binomial count test as the conservative gate before any penalty or reward. Correct for the number of miners: with about 45 UIDs per epoch, use α ≈ 0.01/45 for hard decisions, or report it as a soft score. Never act on the "any batch flagged" rule.
4. **Batch counts:** the window should hold about 6–10 batches of 3 groups per miner, i.e. 18–30 groups, roughly 2–3 of today's 9-group submissions. That gives ≥80% power for a miner adding 1 new solution per batch, and about 3–6 batches for 2 per batch. Today's one batch per epoch means aggregating over about 5 epochs (as in the real-UID check), or asking miners for several small batches per epoch.

**Compute:** 0 GPU-hours (CPU only, about 15 s). Artifacts: `claims/miner_agg.py`, `data/miner_agg.json`, `data/miner_agg.out`.

## v9 adversarial + scalability (Oct 19, 2026, 8:35–11:00 PM CR)

**Question.** Can the proposed v9 rule be gamed, and does it scale? The rule:
- fixed 3-group batches;
- max-group gradient-dot p-value against the epoch's out-of-sample honest-group null;
- per-miner Stouffer + binomial gate at α = 0.01/210 (scored UIDs per epoch);
- 10 batches per miner per window;
- verifier regrade;
- reference drawn per epoch.

**Design under test** (Jacob's 8:53 PM correction). The reference is a *public post-commit random draw from the full env*, seeded by a value revealed only after the commit deadline (e.g. a future block hash). There is no private reference and no secret salt.

**Method.**
- **Simulation.** Simulations run on stored gradient dots (e62/e64/e66; real miner groups as the honest population, 5 random CAL/SIM splits).
- **New GPU runs** (8×B200 pod, shared; live samplers on GPUs 4–7 paused for this, GPUs 0–3 untouched; release scorer jobs untouched; R2 read-only; aggregates only):
  - near-duplicate copies;
  - teacher distillation (066dfb10 as an off-policy teacher at e64);
  - e62 → e64 replay;
  - gradient-norm sketches for every legit and attack group;
  - a no-band-filter reference;
  - 19 real scratch-step batches (Goodhart check).
- **Units.** "Gated" is the fraction of simulated miners passing the v9 gate. **V** is the payment value (sum of clipped z over the miner's 30 groups, in honest-group SD units). The comparison point is a legitimate beyond-policy (BP) miner with one bo16 teacher-correct group per 3-group batch (k=1).
- **Scripts:** `claims/adv_gen.py`, `adv_dots.py`, `adv_prep_replay.py`, `adv_sketch.py`, `adv_scratch.py`, `adv_scratch_spec.py`, `adv_band.py`, `adv_band_an.py`, `adv_cpu*.py`, `adv_final.py [none|cos|cond]`, `adv_pool.py`.
- **Outputs:** `claims/data/adv_final*.json`, `adv_cpu*.json`, `adv_band.json`, `adv_pool.json`, `data/adv/e64/scratch-*.jsonl`.
- **Variability.** Gated rates vary a lot between splits (heavy-tailed honest groups). Ranges are given where they matter. Two full reruns with different random streams moved e64/e66 gated rates by up to ±0.2, so treat differences under ~0.2 as noise. The table uses the saved `adv_final.json` run.

### Verdict per attack

| # | attack | attacker cost | payoff under v9 (gated; V), e62 / e64 / e66 | real held-out effect (scratch, 2 steps, ΔJ×1e4 on held-out half; honest −1.0) | verdict / defense |
|---|---|---|---|---|---|
| ref | legit BP miner, k=1 | bo16 sampling | 0.59 / 0.83 / 0.20; V 20 / 37 / 7 | **+7.0** (11.2, 2.4, 7.5) | target case, works |
| 2a | **Reference leak, live epoch-number seed** (H('ref',E), H('chunks',E) public). Top 3 of 32 honest groups by exact score | ~32× sampling + local scoring, ≈$1.3 GPU per batch | **x32: 0.90 / 0.88 / 0.96, V 71 / 100 / 107. x64: 1.00 / 1.00 / 1.00, V 102 / 168 / 155** | cherry-on-half0 x32: +2.7 held-out (selection transfers but is overpaid ~5× per unit of real gain vs BP) | **Broken. Must fix** with the post-commit public draw |
| 2b / 1 | **Post-commit draw, attacker uses a proxy reference** (independent draw from the same env distribution) | same as 2a + own reference (~1 GPU-h) | x8: 0.02 / 0.02 / 0.08. x32: 0.46 / 0.27 / 0.54 (V 11 / 24 / 19). x64: 0.77 / 0.58 / 0.66 (V 23 / 52 / 28) | as above (+2.7) | Residual and **real** (group-score correlation across independent draws ρ = 0.17–0.32). It is mostly the variance effect in row 6, not leakage |
| 1b | Subject/difficulty-mix guessing under the post-commit draw | free | 0.05 / 0.01 / 0.01, V ≤1 | n/a | Not material |
| 6 | **Variance lottery, reference-free**: top 3 of 3n honest groups by mean length or group gradient norm | n× sampling (+1 backward for gnorm) | length x32: 0.45 / 0.70 / 0.84 (V 14 / 59 / 57); x64: 0.71 / 0.91 / 0.98. gnorm x64: 0.76 / 0.99 / 0.99 | **length x32: +5.8** (6.2, 5.4); real gain, comparable to BP | **Not a fake gain: it is curation.** It is paid comparably to BP per unit of real gain. Policy question (see below) |
| 7 | Easy-task mining (≥75% pass) | free | 0.00 | −1.2 (= honest) | Not material |
| 3 | **Near-duplicate copies** of another miner's BP solutions (e64): hint-paraphrase (base rewrites with the original as a hint; 5-gram Jaccard median 0.07) / prefix-50% copy | needs the victim's data (public after release) + one sampling pass | hint 0.77, V 35; prefix50 0.88, V 43 (BP at e64 0.83, V 37) | **hint copies +4.0** (4.1, 3.9) | Copies are real, useful data, but **paid again**. Exact-hash dedup misses paraphrases. Need task-level first-solver credit (see design changes) |
| 3b | **Replay** of e62 BP groups at e64 | free | **0.94, V 47** (more than fresh BP) | (same data class as BP) | **Paid repeatedly** until the policy learns it. Need a cross-history ledger plus first-solver credit |
| 4 | Sybil splitting, K BP groups over h hotkeys | hotkeys | total V, K=20, e62: h=1 60 · h=2 50 · h=3 19 · h=5 4 · h=10 0.3 (e64: 75 / 51 / 6.6 / 0) | n/a | **Splitting loses**: the gate is concave in evidence. Noise sybils: honest per-hotkey FP ≤7e-4 |
| 5 | **Teacher distillation** (stronger, off-policy model's correct solutions on hard tasks; 6.6% of 2,560 teacher samples correct) | teacher inference | e64 k=1: **0.79, V 37** (= BP) | **+4.6** (5.2, 4.0); real gain | Works as intended (bitter lesson). Policy decision on external teachers (DESIGN Q10) |
| 6b | Gradient inflation via garbage-as-wrong | free | e66: mean z −0.74, ≈0 flags; e62/e64 batches have no reward variance (void) | n/a (zero-variance batch, no update) | Not material |
| 8 | **Task-pool memorization / answer-hinted rationalization** (pool is a fixed public snapshot with answers; see below) | free | not run directly. The closest measured analog is the hint copies above (53% correct, score like BP) | +4.0 (hint analog) | The pool cannot be secret. Score with leave-task-out chunks. A truly held-out task set is needed for the drift alarm |

**Defenses tried against the variance lottery** (6, and the post-commit residual 2b). All on the stored dots, same gate:

| score variant | BP k=1 gated (e62 / e64 / e66) | length x32 | gnorm x64 | live-seed x32 |
|---|---|---|---|---|
| v9 raw (max-group) | 0.59 / 0.83 / 0.20 | 0.45 / 0.70 / 0.84 | 0.76 / 0.99 / 0.99 | 0.90 / 0.88 / 0.96 |
| cosine (score / group grad norm) | **0 / 0 / 0** | 0 / 0 / 0 | 0 / 0 / 0 | 0.64 / 0.67 / 0.75 |
| null conditional on grad-norm decile | 0.25 / 0.21 / 0.00 | 0.17 / 0.27 / 0.19 | 0.55 / 0.60 / 0.55 | 0.88 / 0.82 / 0.93 |
| permutation-studentized group score | ≈0 | — | — | — |
| split-half min / median-of-quarters | partial, cuts BP too | partial | — | — |
| miner-level replication (both halves at √(0.01/210)) | lower | length x32 still 0.70 / 0.60 / 0.63 | — | — |
| self-studentized miner t / Wilcoxon | k=1 ≈0 | — | 0.23–0.60 (e64/e66) | — |

- **No statistic separates legit beyond-policy data from selected honest data.** The legit signal *is* "large gradient aligned with J". Every rule that removes the norm channel also removes BP.
- **The scratch steps settle what this means.** Every high-scoring class raised held-out J in a real update: BP +7.0, length +5.8, distill +4.6, hint copies +4.0, cherry +2.7, against honest −1.0 and easy −1.2 (n = 2–3 batches each; per-batch SD ≈3). We found **no Goodhart break**: the score is not paying for fake gains.
- **What remains are economic problems:**
  - (i) exact-reference leakage overpays relative to real gain;
  - (ii) the same contribution is paid several times (copies, replay);
  - (iii) miners are paid for selection that the validator could do itself from the ~86% of verified groups it never trains on.
- **Caveats.**
  - The per-chunk corr(predicted, actual) in these 2-task-per-chunk scratch runs is 0.29–0.58. That is lower than the 0.76–0.93 measured earlier with full chunks, because the J evaluation is noisier.
  - The realized ΔJ is ~5–10× the first-order prediction for all classes, including a common negative drift on honest batches. Read it as a ranking check, not a calibration.

**Optimizer (Adam proxy, cheap check).** We used one real step with a sign update, U·sign(g) with U=3e-6, as a proxy for Adam's first step. It ran on 8 batches and was compared with the same batches' max-normalized SGD step 1 (held-out ΔJ×1e4).

| batches | SGD | sign |
|---|---|---|
| honest (3) | −2.0, +0.04, −1.75 (mean −1.2) | −0.72, +0.03, −0.48 (mean −0.39) |
| cherry-x32 (1) | +1.43 | +0.18 |
| length-x32 (2) | +1.68, +1.79 | +1.22, −0.21 |
| BP (2) | +5.78, +1.08 | +1.52, −0.47 |

- **Prediction quality is similar under both:** per-chunk corr(pred, actual) 0.43–0.62 for sign vs 0.37–0.58 for SGD.
- **The ordering holds but the margins shrink.** Under the scale-invariant update, everything moves toward honest. One BP batch no longer helps, and length selection is about equal to BP.
- **Reading** (n=1–3 per class, 1 step): the SGD dot probably overstates the advantage of high-norm groups relative to the Adam trainer.
- **Action:** run the linearization audit (§6, alarm 2) in the trainer's actual update form, and consider scoring with a preconditioned dot [U]. A note on an earlier attempt: a first sign run at U=1e-3 was far outside the linear regime (ΔJ +0.05) and is discarded.

### Multiple testing at 0.01/210 (3-group batches)

Batches per miner needed for 80% power:

| rule | k=1 (e62 / e64 / e66) | k=2 (e62 / e64 / e66) |
|---|---|---|
| Stouffer only | 9 / 8 / >12 (0.44 at 12) | 4 / 4 / 9 |
| Stouffer + binomial (v9) | 12 / 11 / >12 | 6 / 6 / 12 |
| BH, q=0.05 (10 contributors among 210) | 6 / 5 / >12 (0.78 at 12) | 3 / 3 / 5 |

- Honest FP at the v9 gate is ≈0 (≤1e-4 per miner).
- BH's realized FDP is 0.01–0.03 on e64/e66 but **0.13–0.16 on e62 (> q)**. On e62 the CAL null's tail is miscalibrated: honest Stouffer at 0.01 flags 1.9–2.4%.
- BH needs a conservative null (pooled over epochs, or a tail-inflated CAL) before it can be used for payment.

### Leakage: (a) live epoch-number seed vs (b) post-commit block-hash draw

- **(a) Fully exploitable.** The live rule `random.Random(H('ref',E))` draws 200 per subject from the 6,746-task pool, samples are seeded `H('ref',E,i,j)`, and chunks use `H('chunks',E)`, with `H = sha256(':'.join(args))[:8] & 0x7fffffff`. All of this is reproducible from public code, the epoch number and the public base checkpoint, so a miner can rebuild the exact reference and select on it (row 2a).
- **(b) The block-hash draw removes exact selection.** The best a miner can do is select on its own same-distribution draw. That transfers only weakly (h0→h1 Pearson 0.32 / 0.17 / 0.19). V at x32 drops from 71–107 to 11–24 at the same sampling cost; x8 gives almost nothing (≤0.08 gated).
- **Subject-mix guessing adds nothing** (≤0.05 gated).
- **What remains at x64** (0.58–0.77 gated) is the same phenomenon as the reference-free variance lottery: any J-aligned large-norm group scores under any draw.
- **Pass-band filter.** The 2–6/8 band is computed from the post-draw reference samples, so it leaks nothing extra.
- **Band power test (e64).** Adding the 376 tasks with 1/8 or 7/8 correct (91 extra chunks; 600 honest groups + 58 BP groups re-dotted):

| reference | BP mean z | BP k=1 gated (B=10, 0.01/210) | honest gated |
|---|---|---|---|
| 2–6/8 (current) | 3.85 | 0.73 | 1e-4 |
| 1–7/8 | 2.07 | 0.58 | 1.5e-4 |
| 1/8 and 7/8 tasks only | 0.87 | 0.12 | 1.5e-4 |

  **Keep the band filter, computed after the draw.** The 1/8 and 7/8 tasks are single-sample contrasts and dilute power.

### Task pool and overlap (Allan's question)

- **What the pool is.** affine_math is a **fixed snapshot**, not procedurally generated: `assets/original-math7496.tasks.json`, with a pinned source_hash.
  - 7,496 distinct competition-math problems: 7 subjects, Levels 1–5, 2,904 distinct answers.
  - **Gold answers ship inside the snapshot**, so they are public.
  - The live index offers 6,746 tasks (90%). The other 750 are a subject-stratified ~10% not offered; their purpose was not determined [U].
  - Tasks cannot be regenerated. The same tasks recur every epoch.
- **Memorizing to cover a random draw.** The post-commit draw takes 1,400 of 6,746 (20.8%); about 24% of those pass the band, i.e. **~340 scored tasks per epoch**. The draw is uniform, so covering a fraction f of a draw's scored tasks needs coverage f of the whole pool: **all 6,746 for certainty** (~1,600 band-eligible at a given base, and which ones drift as the base trains).
  - That is small and public, so memorizing the pool is cheap.
  - **Nothing in v9 may rely on the pool being unknown.**
- **Overlap measured on real miner batches:**
  - submissions cover 1,011 / 1,077 / 1,263 distinct tasks per epoch (15–19% of the pool), and 1,834 (27%) over the three epochs;
  - in a random 1,400-task draw, **20.6%** of submitted groups' tasks are drawn, but only **2.7–3.6%** land in a scored (band) chunk.
- **Does in-sample overlap inflate scores?** Rescoring those groups *leave-task-out* (dropping the chunks that contain the group's own task) changes nothing:
  - mean z −0.06 → −0.06 / −0.05 → −0.07 / +0.58 → +0.67;
  - z>2 rate unchanged;
  - the e66 bump is a mid-difficulty selection effect, not own-task leakage.
- **Recommendation: score leave-task-out; do not exclude submitted tasks from the draw.**
  - Leave-task-out is free and cannot be manipulated.
  - Excluding submitted tasks costs 15–19% of the pool, and it lets a miner steer the reference away from tasks by cheaply submitting groups on them (an exclusion attack).
- **The real memorization risk is train/test contamination.** The trainer trains on the same pool the reference is drawn from, so a rise in J can be pool memorization. The drift alarm needs tasks that are never offered or trained: possibly the 750 non-offered tasks if they are truly never used [U], or another env.
- **Public answers make answer-hinted rationalizations free** (row 8). Measured hint-conditioned solutions score like BP and do raise held-out J (STaR-style). Treat this as legitimate but **pay once per task** (first-solver credit).

### Scalability (GPU cost per epoch of v9 scoring)

**Measured inputs** (B200; MiMo-9B hybrid, `causal_conv1d` *not* installed, so the slow PyTorch fallback is used):
- **Per-rollout gradient dots: ≈8 rollouts/s per GPU** (3,450 real rollouts in 429 s; 2.7–3.1/s for long sequences; dots-only sketches 13/s).
- **Reference per epoch: ≈1 GPU-h**, made of generation (1,400 tasks × 8 samples ≈ 0.76 GPU-h) plus 72–84 chunk gradients (≈0.2 GPU-h).
- **Calibration, gating and aggregation:** CPU, seconds.
- **Linearization audit:** ≈10 GPU-min per batch.

**How long scoring one epoch takes** (wall clock from the commit deadline; reference + dots).
- Model: dots at 8 rollouts/s per GPU; ×2.7 for all-long (≈8k-token) rollouts in parentheses.
- Reference: 58 GPU-min (≈8 min on 8 GPUs, measured 5.7 min generation + 1.5–2 min chunk gradients).
- Load/IO overhead ≈2 min. CPU calibration and gate: seconds. Verifier regrade on CPU runs in parallel with the reference generation.
- [D] from [M] throughputs; not run end-to-end at these loads.

| miners × batches/epoch | rollouts (×24) | dot GPU-h | GPU-h incl. reference | **wall clock, 1 B200** | **wall clock, 8 B200** | fits one epoch (3.6–4 h) on 1 B200? |
|---|---|---|---|---|---|---|
| 100 × 1 | 2.4k | 0.08 (0.23) | 1.1 | 65 min (75) | 10 min | yes |
| 100 × 3 | 7.2k | 0.25 (0.68) | 1.25 | 75 min (100) | 12 min (15) | yes |
| 100 × 10 | 24k | 0.83 (2.3) | 1.8 | 1.8 h (3.2 h) | 16 min (27) | yes |
| 256 × 1 | 6.1k | 0.21 (0.58) | 1.2 | 73 min (95) | 12 min (14) | yes |
| 256 × 3 | 18.4k | 0.64 (1.7) | 1.6 | 1.6 h (2.7 h) | 15 min (23) | yes |
| 256 × 10 | 61k | 2.1 (5.8) | 3.1 (6.8) | 3.1 h (6.8 h) | 26 min (54) | borderline / no |
| 1000 × 1 | 24k | 0.83 (2.3) | 1.8 | 1.8 h (3.2 h) | 16 min (27) | yes |
| 1000 × 3 | 72k | 2.5 (6.8) | 3.5 (7.8) | 3.5 h (7.8 h) | 29 min (61) | borderline / no |
| 1000 × 10 | 240k | 8.3 (22.5) | 9.3 (23.5) | 9.3 h (23.5 h) | 72 min (3 h) | no: needs 3–6 B200 (8 to finish within ~1–3 h) |

Add the linearization audit, 6 batches × ≈10 GPU-min ≈ 1 GPU-h/epoch (8 min on 8 GPUs). It is off the critical path.

**Reading.**
- Today's load (≈210 UIDs × ≤3 batches) is about 15 min of scoring on the 8-GPU pod, or ≈1.6 h on one B200.
- The fixed ≈1 GPU-h reference dominates up to ~256 × 3.
- From ~1000 × 3, per-rollout backward passes dominate and one B200 no longer finishes inside an epoch if rollouts are long.

**Bottlenecks and what changes.**
1. **The post-commit draw puts the reference on the critical path.** It can only be generated after the deadline: ≈8 min on 8 GPUs, ≈60 min on 1.
2. **Per-rollout backward passes dominate.**
   - Production scoring needs only 2–3 combined reference vectors (full / half0 / half1, 1.2 GB each), not the 101 GB per-chunk matrix. Per-chunk dots are diagnostics only.
   - Installing causal_conv1d and packing sequences should give an estimated 3–5× [U].
3. **Validator-side curation is cheap.** Grad-norm sketches cost the same backward pass, which helps the policy question on the variance lottery.
4. **Envs beyond affine_math:**
   - reference generation and chunk gradients scale per env;
   - **nulls must be per env** (length and grad-norm distributions differ);
   - non-math verifiers (code sandboxes) add latency on the critical path;
   - the multiple-testing count grows with any per-env testing;
   - the band filter and the leave-task-out rule carry over unchanged.
5. **Cost** at the pod rate ($71.12/h for 8 GPUs = $8.89/GPU-h): 256 × 3 ≈ $14/epoch (≈$85/day); 1000 × 10 ≈ $83–200/epoch.

### Required design changes

1. **Post-commit public draw from the full env** (future block hash), replacing `H('ref',E)`. Reference generation, band filter and chunks are all computed after the draw. **Blocking**: the live seed is fully exploitable.
2. **Pay once per contribution.**
   - Task-level first-solver credit: within a window, credit for beyond-honest scores on task i goes to the earliest commitment. Later batches on the same task are scored but paid only if they beat it.
   - A cross-history ledger of (task, completion hash) and of paid tasks, to stop replay and re-payment of released data.
   - Exact-hash dedup alone misses paraphrases (5-gram Jaccard 0.07 still pays fully).
3. **Leave-task-out scoring** (drop the chunks that contain the group's task). Do **not** exclude submitted tasks from the draw.
4. **Decide the curation policy** before go-live (the variance lottery). Measured: selecting long or high-gradient honest groups gives real held-out gains and is paid like BP. The options:
   - (a) accept it as paid curation;
   - (b) make the validator curate itself: train on the top-scoring verified groups from the 86% surplus, and pay only for scores above what validator-side selection from the honest surplus achieves (e.g. a null built from the top decile of honest groups) [P][U].
   Grad-norm normalization and conditional nulls are **rejected**: they kill legit power.
5. **Window ≥ 12 batches per miner** for k=1 at 80% power with the both-gate on e62/e64-like epochs. e66-like epochs need k≥2 or BH. BH is only allowed with a conservative null (e62 FDP 0.13–0.16).
6. **Keep the 2–6/8 band filter.**
7. **Held-out drift set outside the trained pool** (contamination guard).

### Compute and cost for this round

- About 9.5 GPU-h:
  - generation (copies, distill) ≈0.8;
  - dots ≈0.5;
  - sketches ≈0.7;
  - band ≈0.6 (+0.2 lost to two OOM restarts);
  - scratch steps ≈5.5 (incl. ≈1 GPU-h lost to a first batched attempt);
  - Adam-proxy runs ≈1.0 (incl. a discarded U=1e-3 run).
- CPU simulations: minutes.
- Pod `affine-sampler-8xb200` at $71.12/h since 10:53 AM CR: ≈$860 cumulative at 11:00 PM CR. This round (8:35–11:00 PM) ≈$172.

**State of the pod.** All adversarial GPU jobs are finished. The trusted-sampler instances on GPUs 4–7, which were paused at ~8:38 PM for this work, were relaunched at ~10:46 PM CR (`logs/launch-live-g47-after-v9adv.out`, staggered 60 s). GPUs 0–3 and the release scorer were not touched.
