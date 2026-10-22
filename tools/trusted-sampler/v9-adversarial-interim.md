# v9 adversarial + scalability — INTERIM (Mon Oct 19 2026, ~9:15 PM CR)

> **Superseded (11:00 PM CR)** by the final "v9 adversarial + scalability" section of REPORT-claims.md and the merged DESIGN-v9.md. Some interim numbers, from an earlier simulation run, differ by up to ±0.2.


The whole simulation uses stored gradient dots for real miner groups (e62/e64/e66), with 5 random CAL/SIM splits per epoch.
- **v9 test as specified:**
  - 3-group batches; batch statistic = max-group p-value against the CAL honest null.
  - B=10 batches per miner.
  - Gate: Stouffer AND binomial test, both at alpha = 0.01/210.
  - Payment V = sum of clipped z over the miner's 30 groups, in honest-group SD units.
- **Reference point:** a legitimate "beyond-policy" miner (1 bo16 teacher-correct group per batch, k=1) is gated 59% / 75% / 18% (e62/e64/e66) with V = 20.5 / 31.8 / 6.3.
- **Ranges:** values are means over the splits, with [min, max] shown where the spread is large.
- **Constraints kept:** aggregates only, miner data private, R2 read-only, release scorer jobs and pinned code not touched.

## Threat table (measured unless marked)

| # | Attack | Attacker cost | Payoff under v9 (gated rate; V) e62/e64/e66 | Detection | Proposed defense |
|---|---|---|---|---|---|
| 2a | **Reference leak, live seed rule** (H('ref',E)/H('chunks',E) are public, so the exact reference is reproducible). Sample 32 honest groups per slot, submit the top 3 by exact score. | ~32× sampling plus 1 local scoring pass (≈ $1–3 per window) | x8: 0.13/0.05/0.16. **x32: 0.90/0.72/0.91, V 71/70/96. x64: 1.00/0.99/1.00, V 102/138/145** (3–20× a legit miner) | None: the groups are genuine honest samples | **Mandatory**: draw the reference after the commit deadline from a public, unpredictable seed (future block hash) over the full env. |
| 2b / 1 | **Post-commit draw; attacker uses a proxy reference** from an independent draw of the same env distribution (selection transfer). | Same as 2a, plus building its own reference | x8: 0.02/0.00/0.02. x32: 0.46/0.22/0.38 (V 11/14/10). x64: 0.77/0.53/0.52 (V 23/30/15) | None | The block-hash draw cuts the leak by ~3–6× in V, but **does not eliminate it**: group-score correlation across independent reference draws is ρ≈0.17–0.32 (Spearman 0.11–0.32). |
| 6/7 | **Reference-free variance lottery**: top 3 of 3n honest groups by mean completion LENGTH or group gradient norm; needs no reference at all. | ~n× sampling; gnorm needs one backward pass | length x32: 0.45/0.50/0.69 (V 14/32/40); x64: 0.71/0.84/0.93 (V 28/80/76). gnorm x64: 0.76/0.97/0.96 (V 29/110/79) | None. Spearman(gnorm, \|z\|) = 0.40–0.52 | **Open.** Tested and failed: permutation-studentized score (kills legit signal too), split-half min/median, both-halves replication, self-studentized miner t / Wilcoxon. Now testing a norm-normalized (cosine) score and a gnorm-conditional null. Scratch steps (running) will show whether these batches really raise held-out J. If they do, this is real data curation, not an exploit. |
| 1 | Subject/difficulty guessing (post-commit) | ~free | gated 0.05/0.00/0.00 (V ≤1) | n/a | Not material. Stratify the null by subject as a cheap safeguard. |
| 7 | Easy-task mining (≥75% pass) | free | 0.00 | n/a | Not material (A-weighted groups with low variance score ≈0). |
| 3 | **Near-duplicate copies** of another miner's beyond-policy solutions: prefix-copy (50%), or hint-paraphrase where the base re-writes it (5-gram Jaccard 0.07). | Needs the victim's data (public after release) plus 1 sampling pass | hint k1: gated 0.75, V 31.5. prefix50 k1: 0.86, V 39.9. Legit BP at e64: 0.75, V 31.8. Copies pay like originals. | Exact-hash dedup misses them (hint jac5 median 0.07) | **Task-level first-solver credit**: credit for "first correct solution on task i" in a window goes to the earliest commit, and later submissions on the same task get the honest-group null only. Also a global ledger of solved (task, answer) pairs. |
| 3b | **Replay** of old/released beyond-policy groups (e62 data re-scored at e64) | Free | gated 0.94, V 43.7 (more than fresh e64 BP) | Exact hash catches verbatim replay only if a cross-epoch ledger exists | **Cross-history (task, completion-hash) ledger**, plus "already solved by policy or ledger" means no credit. |
| 4 | Sybil splitting (K BP groups over h hotkeys) | Hotkey cost | Total V for K=20, e62: h=1 60, h=2 50, h=3 19, h=5 4, h=10 0.3. **Splitting loses.** | Honest per-hotkey FP ≤ 7e-4 (both-gate) | None needed for splitting. The gate is concave in evidence, so concentration wins. |
| 5 | Teacher distillation (066dfb10 teacher, e64) | Teacher inference | **Pending** (generation done, dots running) | | |
| 6b | Gradient-norm inflation via garbage-as-wrong | Free | e66 7 groups: mean z −0.74, flags ≈0. e62/e64: no reward variance | n/a | Not material (verifier regrades). |

## Multiple testing at 0.01/210 (3-group batches, k = legit groups per batch)

Batches needed for 80% power:

| rule | k=1 (e62 / e64 / e66) | k=2 (e62 / e64 / e66) |
|---|---|---|
| Stouffer only | 9 / 8 / >12 (0.44 at 12) | 4 / 4 / 9 |
| Stouffer + binomial (v9) | 12 / 11 / >12 | 6 / 6 / 12 |
| BH q=0.05 | 6 / 5 / >12 (0.78 at 12) | 3 / 3 / 5 |

- Honest FP at the both-gate is ≈0.
- BH realized FDP: e64/e66 0.01–0.03, but **e62 0.13–0.16 (> q)**. The cause is null-tail miscalibration: honest Stouffer at 0.01 is 1.9–2.4% on e62.
- BH needs a conservative null (pooled across epochs or tail-inflated) before use.

## Leakage summary, (a) vs (b)
- (a) The live epoch-number seed is fully exploitable (row 2a).
- (b) The post-commit block-hash draw drops the gain to proxy-transfer level (row 2b).
- The residual (b) gain is the same order as the reference-free variance lottery (row 6/7), i.e. not reference leakage as such: groups with large J-aligned gradients score high under any draw.
- The pass-band filter is computed on the post-draw reference samples, so it leaks nothing. The power test without the band filter is running.

## Scalability

Inputs:
- Dots: 8 rollouts/s per B200 measured (forward through the full model, backward through layers 28–31; 2.7–3.1/s for long sequences).
- Fixed per epoch: post-commit reference ≈ 1 GPU-h (generation 0.76 + chunk gradients 0.2). This is on the critical path after the deadline: ~8 min on 8 GPUs, ~60 min on 1.
- Calibration and gating: CPU, seconds.

GPU-hours per epoch:

| miners × batches/epoch | rollouts (×24) | dot GPU-h | + fixed | 1 B200 enough? (epoch ≈3.6–4 h) |
|---|---|---|---|---|
| 100×1 | 2.4k | 0.08 | 1.1 | yes |
| 100×3 | 7.2k | 0.25 | 1.25 | yes |
| 100×10 | 24k | 0.83 | 1.8 | yes |
| 256×1 | 6.1k | 0.21 | 1.2 | yes |
| 256×3 | 18.4k | 0.64 | 1.6 | yes |
| 256×10 | 61k | 2.1 | 3.1 | borderline (long sequences ×2.7 → no) |
| 1000×3 | 72k | 2.5 | 3.5 | borderline |
| 1000×10 | 240k | 8.3 (up to 22 for long sequences) | 9.3+ | no: 3–6 B200 |

Bottlenecks and fixes:
- The per-rollout backward pass dominates.
- The full per-chunk V matrix (84 × 604M bf16 = 101 GB) is needed only for diagnostics. Production scoring needs only 2–3 combined vectors (full / h0 / h1 = 1.2 GB each).
- causal_conv1d is not installed (slow fallback) and there is no sequence packing. An estimated 3–5× speedup is available [untested].
- More envs: reference generation, chunk count and null calibration scale per env.
  - Separate per-env nulls are needed, since length and gnorm distributions differ.
  - Non-math verifiers (code sandboxes) add CPU/latency on the critical path.
  - Multiple-testing count × #envs if tested per env.
- Cost at $8.89/GPU-h: 256×3 ≈ $14/epoch (~$85/day); 1000×10 ≈ $83–200/epoch.

## Required design changes so far
1. Replace the epoch-number seed with a **public post-commit random draw from the full envs** (block hash after the deadline). Build the reference and pass rates after the draw.
2. **First-solver credit plus a cross-history ledger** for (task, answer) and (task, completion hash). This is needed against paraphrase copies and replay.
3. **The variance lottery is unresolved**: v9 pays ~n× sampling selection on length/gnorm about as much as genuine beyond-policy data. Do not go live until either (a) scratch steps show these batches really move held-out J (then it is legitimate curation and the payment is fair), or (b) a normalized or conditional score removes it while keeping legit power.
4. Budget **≥10–12 batches per miner per window** for 80% k=1 power at 0.01/210 with the both-gate (e66-like epochs: not reached at 12). Or use BH with a conservative null.

## Still running
- Scratch-step Goodhart check: 17 batches, 2 SGD steps, ref vs held-out J.
- Distill dots.
- Band-filter power.
- Cosine / conditional-null defenses.
- Adam-sign vs SGD.

GPU use this task so far ≈ 3 GPU-h. Pod cost since 10:53 CR ≈ $730 at 9:15 PM.
