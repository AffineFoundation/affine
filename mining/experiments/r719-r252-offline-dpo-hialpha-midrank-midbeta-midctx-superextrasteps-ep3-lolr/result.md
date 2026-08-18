# R719 — MidCtx MidRank MidBeta SuperExtra (r252)

**Axis:** Offline-DPO HiAlpha MidRank MidBeta MidCtx SuperExtraSteps ep3×LoLR
β=0.1 r=32 α=128 lr=1e-6 @8192 max_steps=**14400**

**Decision rule:** Stage-5 iff fresh v4 n80 `margin > max(2·SE, δ=0.002)` AND median `|z|≥80` AND B pass ≥0.30 vs **reign35** (wvk=7, k=3, τ=0.03).

## Status (p3768)
**N80 LIVE** on `mine-r262` (golden) GPUs **4,5**/:8002 vs reign35 · vllm**537007** sim**542450** · `*_reign35_wvk7` fail-closed k=3 · MERGE_DONE `/tmp/r719_merged` 66G/16sh · hard-pin GPUS after mine.env.
