# SN120 trusted paired sampler: one-pass, optimization, CRN, log-prob prototype and per-miner credit backtest

Generated 2026-10-19 on `affine-sampler-8xb200` (Lium `calm-hawk-34`, 8×B200). Model: MiMo-V2.6-Distill-Qwen-9B
mainnet run `mimo-mainnet-v1`. Everything here is read-only with respect to the network. Miner data appears only as
aggregates; UIDs are the only identifiers used (and none are listed).

Step numbering: *step s* = the checkpoint produced by training epoch s−1 (its `optimizer_step`). Pair "step s" =
(step s−1 → step s), i.e. the effect of training epoch s−1.

## 1. Per-step paired deltas (backfill, raw correctness)
Every record draws its own uniform K tasks from the 6,746-task pool and runs m=8 samples per task on each side.
Task-level pooling across records; this table is CRN-only (no-CRN comparison records are excluded).

| step | pair | records | tasks | before | after | delta | paired SE | unpaired SE | z |
|---|---|---|---|---|---|---|---|---|---|
| 60 | 05107c41→2198992b | 7 | 896 | 0.732 | 0.730 | −0.0014 | 0.0049 | 0.0162 | −0.28 |
| 61 | 2198992b→4e7d422e | 7 | 896 | 0.733 | 0.732 | −0.0006 | 0.0048 | 0.0162 | −0.12 |
| 62 | 4e7d422e→c8a6d342 | 7 | 896 | 0.747 | 0.751 | +0.0045 | 0.0047 | 0.0157 | +0.95 |
| 63 | c8a6d342→8119f5e2 | 7 | 896 | 0.753 | 0.759 | +0.0060 | 0.0050 | 0.0156 | +1.21 |
| 64 | 8119f5e2→cf791667 | 7 | 896 | 0.775 | 0.775 | +0.0001 | 0.0048 | 0.0149 | +0.03 |
| 65 | cf791667→19f21658 | 10 | 1664 | 0.752 | 0.755 | +0.0029 | 0.0036 | 0.0115 | +0.82 |

Length-penalized reward (live overlong penalty), same records:

| step | delta | paired SE | unpaired SE |
|---|---|---|---|
| 60 | −0.0008 | 0.0060 | 0.0245 |
| 61 | +0.0004 | 0.0055 | 0.0241 |
| 62 | +0.0050 | 0.0058 | 0.0237 |
| 63 | +0.0099 | 0.0060 | 0.0233 |
| 64 | +0.0007 | 0.0056 | 0.0224 |
| 65 | +0.0038 | 0.0043 | 0.0173 |

* The per-epoch delta is noisy: the mean paired SE is ≈0.0046 at ~900 tasks (≈0.013 per K=128 instance and ≈0.009
  per K=256 instance). The precision-weighted mean across the 6 steps is **+0.0020** (SE ≈0.0018). The
  between-epoch SD of the point estimates (0.0030) is *smaller* than the mean SE, so there is no detectable
  epoch-to-epoch variation in true improvement. Steps are individually indistinguishable from 0 (|z| ≤ 1.6).
* Pairing cuts SE by ≈3.3× (variance ≈10×) compared with unpaired estimates on independent draws: per-task
  before/after correlation ≈0.89–0.91.
* Steps 54–58 could not be backfilled. Their presigned GET URLs had expired, the checkpoints are not cached on any
  pod we may use, and the only R2 credentials available are account-wide read-write keys, which the rules forbid.

## 2. Common random numbers (CRN) vs independent seeds, K=128, m=8
**(i) Separate runs (different task draws).** Paired SE per K=128 instance, from the task-level SD of the paired difference:

| step | CRN: records / tasks / SE@K128 / corr(before, after) | no CRN: records / tasks / SE@K128 / corr |
|---|---|---|
| 62 | 7 / 896 / 0.0125 / 0.909 | 5 / 640 / 0.0133 / 0.895 |
| 63 | 7 / 896 / 0.0131 / 0.899 | 5 / 640 / 0.0124 / 0.913 |

**(ii) Matched combined experiment (same 128 tasks per run; 4 runs × K=128 per pair, all 8 GPUs).** See §3.

CRN gives at most a small gain. In the matched design it cuts SE by 7–11% (variance×cost 0.79–0.87 of baseline);
in the unmatched runs the difference is within noise (±6%). The reason: before and after share seeds, but once the
weights differ, the ~2,900-token trajectories diverge within a few tokens, so outcome correlation rises only
0.886→0.898 (step 63) and 0.895→0.916 (step 65). CRN is free, so it stays on by default. It is not a large lever.

## 3. Combined experiment: CRN × log-prob scoring
Four-way comparison, same tasks within each run, K=128 per run, 4 runs per pair (512 tasks per pair). SE is quoted per
K=128. GPU-min is per K=128 run on one B200, including the cold engine load (~90 s).

| step 63 (c8a6d342→8119f5e2) | delta | SE@K128 | GPU-min | var×cost vs baseline |
|---|---|---|---|---|
| (a) baseline, independent seeds | −0.0034 | 0.0136 | 7.8 | 1.00 |
| (b) CRN only | +0.0034 | 0.0127 | 7.8 | 0.87 |
| (c) log-prob only (before samples, IS-reweighted to after) | **+0.0435** | 0.0134 | 6.5 | 0.80 |
| (d) CRN + log-prob (min-var combo of CRN, fwd-IS, rev-IS) | **+0.0260** | 0.0082 | 11.4 | 0.53 |

| step 65 (cf791667→19f21658) | delta | SE@K128 | GPU-min | var×cost vs baseline |
|---|---|---|---|---|
| (a) baseline, independent seeds | +0.0022 | 0.0136 | 7.9 | 1.00 |
| (b) CRN only | +0.0090 | 0.0121 | 7.9 | 0.79 |
| (c) log-prob only | **+0.0325** | 0.0129 | 6.6 | 0.75 |
| (d) CRN + log-prob | **+0.0260** | 0.0080 | 11.5 | 0.51 |

Mean phase times per run: load 90 s, generation ≈177 s per side, verification ≈1 s, each prefill log-prob pass
≈48 s (≈1.8M tokens), swap ≈22 s.

**The log-prob estimators are biased, so (c) and (d) are not usable as they stand.** They read +0.03 to +0.04 on
pairs whose direct paired deltas are +0.003 to +0.006 (and +0.0029 ± 0.0036 over 1,664 tasks for step 65). The
reverse direction is biased the same way (+0.040, +0.056). So the smaller SE of (d) is real, but it measures a
biased quantity. Likely causes:
1. The weights use untruncated-softmax log-probs (vLLM `prompt_logprobs`), while sampling used top-k=20 /
   top-p=0.95. That makes the proposal ratio wrong.
2. The sequence-level log ratio is strongly length-dependent: mean −1.9 nats/sequence, ≈−0.0006/token. Long
   samples, which are more often wrong or unresolved, get systematically down-weighted.
3. Self-normalized IS over 8 samples per task is biased (median ESS ≈4.8 of 8).

Jacob's original length-normalized score, s = mean dlogp/len(correct) − mean dlogp/len(wrong), is computable on
≈51% of tasks (those with both outcomes). It gives +5.8e-5 ± 4.0e-5 (z 1.4) for step 63 and +8.2e-5 ± 3.8e-5
(z 2.2) for step 65. In the standalone prototype it gave +2.0e-5 ± 8.7e-5 (step 62) and three step-63 runs of
−1.0e-5, +8.1e-5 and −1.6e-4 (SE ≈6–10e-5). It is in nats/token, has no calibrated null, and may carry the same
length confound, so its z cannot yet be compared with correctness z. What it costs: 1 generation plus 2 prefill
passes (~6.5 GPU-min vs ~7.8 per K=128 run; in chain steady state ≈295 s vs ≈355 s per K=128 pair, ≈17% cheaper).

**Next step if pursued:** recompute the weights with processed (top-k/top-p-truncated, temperature-applied)
log-probs (vLLM `logprobs_mode='processed_logprobs'`), and calibrate on a placebo. Until then, use (b) CRN
generation scoring.

## 4. Per-miner credit regression
**Data.** Epochs 59–64 (6 steps) with paired deltas. For each epoch, the trained groups come from the
TRAINING-JOB documents, and pubkey→UID comes from the audit population: 256 groups per epoch, 214 distinct UIDs,
≈144 UIDs per epoch, each UID present in ≈4.0 of 6 epochs. Mean share when present is 0.0069 (max 0.0195, i.e. 5
groups).

**Model.** delta_e = τ0 + Σ_i z_ie·u_i + ε_e, where z_ie is UID i's share of epoch e's 256 groups, ε_e ~ N(0, SE_e²)
from the sampler, and u_i ~ N(0, s_u²) is ridge shrinkage. A miner's credit is τ_i = τ0 + u_i: the improvement a
whole batch of that miner's groups would give. Because Σ_i z_ie = 1, τ0 absorbs the common level, and only
deviations between miners are estimable. s_u was chosen by marginal likelihood (empirical Bayes), with fixed
s_u ∈ {0.01, 0.05, 0.25} as checks.

| quantity | result |
|---|---|
| noise of per-epoch delta | paired SE ≈ 0.0046 (≈900 tasks); between-epoch SD of estimates 0.0030 |
| empirical-Bayes s_u | at the lower bound (1e-4): the data prefer *no* miner heterogeneity |
| UIDs with \|τ_i − τ0\| > 2 posterior SD | **0 / 214** at every s_u (max \|z\| 0.11) |
| posterior SD / prior SD (median) | 0.99–1.00, so essentially nothing is learned per miner |
| split-half rank stability (fit on alternate epochs, 192 UIDs in both halves) | Pearson ≈ −0.11, Spearman ≈ −0.07 to −0.10 (≈0) |
| would-be pay (τ_i⁺ normalized, s_u=0.25) vs current never-burn assessment (211 UIDs) | Spearman 0.007; total-variation distance 0.50; Gini 0.63 vs 0.20; top-10 overlap 0/10 |
| correlation with share of trained groups | current pay 0.64, would-be pay 0.02 |

**Power (simulation, fresh random-256 designs drawn with each UID's observed share).** The true per-miner
heterogeneity SD is s_u, in units of "delta if the whole batch were this miner's".

| true SD of u | epochs | per-epoch SE | UIDs with \|z\|>2 | corr(true, estimated) |
|---|---|---|---|---|
| 0.01 | 6 | 0.0046 | 0% | −0.03 |
| 0.01 | 100 | 0.0046 | 0% | 0.06 |
| 0.01 | 1000 | 0.0046 | 0% | 0.24 |
| 0.01 | 1000 | 0.001 | 11% | 0.76 |
| 0.05 | 6 | 0.0046 | 0% | 0.06 |
| 0.05 | 100 | 0.0046 | 0% | 0.39 |
| 0.05 | 1000 | 0.0046 | 14% | 0.79 |
| 0.05 | 1000 | 0.001 | 72% | 0.98 |

**Reading.** u is in units of "delta if the whole batch were this miner's" (the mean step gain is ≈0.002, so
s_u=0.01 means miners differ by ~5× the mean gain). At the current noise (SE ≈0.0046 per epoch, ≈7 instance-K128 runs) per-miner credit is not
identifiable. That holds even after 1,000 epochs, unless miners differ by ≥5× the mean step gain *and* the
per-epoch SE is cut to ≈0.001 (≈20k paired tasks per epoch, roughly 16 B200-GPU-hours per epoch at the measured
throughput). The would-be pay distribution from these 6 epochs is noise: it is more concentrated than current pay
only because of the τ⁺ truncation, and uncorrelated with both current pay and volume.

**Confounding (important).** Inclusion was *random-256-groups* from the submitted pool, not an independent coin
flip per miner:
1. Shares are compositional (they sum to 1), so a miner's effect is only identified relative to the others in the
   same batch, and it is collinear with everyone's volume.
2. Share tracks submission volume (and first-claim timing), so z is correlated with miner size and age. That is
   why current pay correlates 0.64 with share. Any τ estimate mixes "this miner's data quality" with "how much this
   miner submitted".
3. Epoch-level drift (learning-rate schedule, model state, task mix, a different pool snapshot) is shared by all
   miners in an epoch and is indistinguishable from the batch composition, with only 6 epochs.
4. Groups within an epoch interact (the GRPO update is not additive in groups), so the linear model is only an
   approximation.

Identifying per-miner contributions would need **randomized per-miner ablations**: for example, a held-out
counterfactual step with miner i's groups dropped (or randomized on/off per miner independently), evaluated paired on
the same tasks. A pooled paired delta per step is well suited for monitoring "is training helping"; it is not
suited for per-miner pay.

## Files
`/root/trusted-sampler/` on the sampler pod (mirrored at `~/subnet120-rewrite/tools/trusted-sampler/` on Arbos):
`results/*.jsonl`, `data/pooled.json`, `data/backtest.json`, `data/inclusion.json` (UID aggregates),
`results-logprob/*.jsonl`, `results-combined/*.jsonl`, `data/combined.json`, `backtest.py`, `crn_compare.py`,
`logprob_proto.py`, `combined.py`, `combined_analyze.py`.
