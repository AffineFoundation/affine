**wvk 15 → 16 is live: the per-duel crown rule is back — margin > max(2·SE, δ = 0.002). Reign 13 is uncrowned; reign 12 stands.** Explicit dated operator directive 2026-09-13 12:10 UTC. Effective from the first duel dispatched after the eval pod redeploy at 13:01 UTC today (`chal-00470` was the first). Forward-only; no other verdict is re-decided; `min_submission_block` unchanged.

**What happened.** Reign 13 (`chal-00461`, crowned 09:59 UTC today) was a re-upload of reign 12's weights: all 1,026 tensors byte-identical; only the file split differed (16 shards → 2), so the file hashes and model digest changed and the copy check did not see it. Under the 12-hour-window rule it won window 2515 on a margin of +0.0007 (z = 0.93); its confirmation slice was negative; the pooled margin was +0.00005 and the rule crowned on "pooled margin > 0". That is noise, not an improvement — the δ bar refuses it.

**What changes now.**
• `weight_version_key = 16`. A challenger crowns iff its paired margin over one seeded 1,300-turn slice is > max(2·SE, 0.002), plus the unchanged thought-length floor and B gate — the rule of wvk 3–14. The 12-hour window, best-of-window, pooled confirmation slice and near-miss second slice are off. min(R, G) scoring is unchanged.
• Reign 13 is uncrowned (`chal-00461` → `rejected_model_copy`; its crown row is marked revoked; window 2516 closed with no crown). Reign 12 (`king-d76150805915`) stands; weights point at it since 13:14 UTC.
• Re-uploading a crowned model's weights — re-sharded, renamed, or with a few values nudged — will not crown: it sits at the noise floor, below δ, and it burns the hotkey's slot.

Spec: https://affine.io/llms.txt → "Fork history: wvk 16". Knobs: `code/affine.toml [duel] crown_mode`, `near_miss_enabled`, `min_margin`, `k_sigma`.
