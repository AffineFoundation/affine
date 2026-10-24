# SN120 live miner scoring: v9 measured impact, window-credit-v1

Status: **live on Finney netuid 120, mechanism 0, from the 12:00 CR (18:00 UTC) hourly weight run on 2026-10-22.**
This page describes the rule that sets the weights. Older reward documents (continuous audit penalties,
six-hour contribution half-life, always-pay top-10 softmax, share-EMA) are history and do not apply.

## Audit it yourself

| what | where |
| --- | --- |
| Latest scores, per UID, with every input | https://affine.io/scores.json |
| Snapshot for each scored epoch | https://affine.io/scores/e<EPOCH>.json (list: https://affine.io/scores/index.json) |
| Miner guide | https://affine.io/llms.txt |
| Current epoch / upload window | https://affine.io/mining.json |

The validator rewrites the scores files after every hourly writer run. Each file contains:

- `formula`: rule name, config sequence, window, floor parameters and a plain-text formula.
- `score_epoch`, `window` (epochs used and skipped), `seed` (block number + hash + derived seed), `reference`, `null`.
- `miners[]`: per UID `uid`, `hotkey` (ss58), `window` {groups, units, hits, credit}, `by_epoch` {groups, units, hits, credit},
  `newest_epoch_groups`, `zero_groups_floor_ok`, `submitted_and_trained_by_epoch`, `v9_gate_diagnostic` {B, hits, stouffer_z,
  binom_p, gated}, `share_now` (what the rule outputs now) and `share_last_submitted` (the last vector sent to the chain).
- `audit`: label-shuffle and UID-shuffle placebos, unit hit rate vs expected, flag rates and diagnostic alarms.
- `per_task_claims`: for each epoch in the window, task index -> {groups, miners, trained}.
- `epoch_status`: current epoch from mining.json, scoring state per recent epoch (scored / scoring / failed_retrying),
  install time and ETA, plus the median lag from deadline to score.

Only aggregates are published, never rollout text.

## Formula

1. **Inputs.** All groups admitted to the sealed epoch's training documents and built on the epoch's base checkpoint.
2. **Units.** Fixed 3-group units by slot: slots 0-2, 3-5 and 6-8, so at most 3 units per miner per epoch. A partial unit
   (1-2 groups) is tested against the same 3-group null, so splitting never helps.
3. **Reference (public post-commit seed).** `seed_block` = first finalized block with timestamp >= upload deadline, + 10.
   `seed = sha256("sn120-v9-ref-seed-v1:" + epoch_id + ":" + seed_block_hash)`. 200 tasks per subject come from the full offered
   pool (~1,400). Each gets 8 base samples, and the 2-6/8 band filter is applied. The kept tasks form 4-task subject chunks in two halves.
   See `docs/v9-reference/seed_block.py` and `v9common.py`.
4. **Group score.** `s_g = 0.5 * sum_half mean_{chunks k in half, task(g) not in k} (A_g . D_g[:,k]) / (8192 n)`: the
   advantage-weighted per-rollout gradient dot product with the reference chunk gradients of J, in the MLP weights of layers 28-31,
   leaving out the group's own task (`v9agg.group_score`).
5. **Null and p.** A 2-fold cross-fit by miner (`fold = HS(seed,'fold',miner) % 2`) gives each unit an empirical max-of-3 null
   built only from the other fold. `p_u = P_null(max of 3 >= max_g s_g)`, with a half-count continuity correction so that
   0 < p < 1 (`v9agg.Null.p`).
6. **Credit.** `credit(unit) = groups_in_unit * (1 - p_u)`. An average honest full unit earns ~1.5; a unit far above the null earns up to 3.
7. **Window.** `credit_i` = sum over miner i's units in the 4 sealed epochs ending at the newest scored epoch E. Epochs
   without a valid signed score are skipped and never estimated (`weights_v9.window_credit`).
8. **Zero-groups floor.** A miner gets exactly 0 unless it has >= 1 verified group in the window **and** >= 1 in epoch E.
9. **Weights.** `share_i = credit_i / sum_j credit_j` over qualifying miners. 100% of mechanism 0 goes to miners.
   Weight for hotkeys that are no longer registered is redistributed to the rest. There is no top-K, no EMA and no coldkey grouping.
10. **Fail-safe.** If the newest valid score is more than 2 sealed epochs old, or nobody qualifies, the whole weight goes to UID 0 (burn).

The Stouffer/binomial Bonferroni gate (`v9agg.gate_window`) and the placebos are still computed and published, but only as
diagnostics.

## Why it changed (2026-10-22)

The previous rule (seq5/seq6) paid a softmax over the top 10 miners by window Stouffer z, smoothed by a 12-epoch share-EMA.
Because z comes from a 4-epoch window and the EMA carries old shares, miners with no groups in recent epochs were still paid.
Most miners who sent the full 9 groups got nothing, and a few coldkeys with many UIDs held most of the weight. A scorer bug
(p = 1 -> Stouffer -inf -> invalid JSON) also left epochs 85 and 87 unscored until it was fixed and they were re-scored on 2026-10-22. Backtest on epochs 80-88 (overlap = sum of min(weight share, share of verified groups in the newest epoch)):

| rule | paid UIDs | max share | top-2 coldkeys | full-9 miners unpaid | paid with 0 groups in newest epoch | overlap with group counts |
| --- | --- | --- | --- | --- | --- | --- |
| seq6 (old) | 18-28 | 9-28% | 44-92% | 18-60 per epoch | up to 4 | 0.14-0.21 |
| window-credit-v1 | 132-190 | 1.3-2.2% | 20-34% | 0 | 0 | 0.76-0.81 |

## Reference code

`docs/v9-reference/` holds verbatim copies of the live files: scorer aggregation `v9agg.py`, scorer helpers `v9common.py`,
seed rule `seed_block.py`, weight rule `weights_v9.py` and publisher `public_scores_v9.py`. The validator verifies their sha256
from a release signed by the subnet authority key 3301134b...0bcd.
