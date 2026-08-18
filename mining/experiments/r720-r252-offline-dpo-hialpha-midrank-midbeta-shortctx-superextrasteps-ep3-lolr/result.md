# R720 — Short MidRank MidBeta SuperExtra (r252)

**Axis:** Offline-DPO HiAlpha MidRank MidBeta ShortCtx SuperExtraSteps ep3×LoLR
β=0.1 r=32 α=128 lr=1e-6 @6144 max_steps=**14400**

**Decision rule:** Stage-5 iff fresh v4 n80 `margin > max(2·SE, δ=0.002)` AND median `|z|≥80` AND B pass ≥0.30 vs **reign35** (wvk=7, k=3, τ=0.03).

## Status (p3768)
**N80 LIVE** on `mine-r262` (golden) GPUs **6,7**/:8003 vs reign35 · vllm**537020** sim**542323** · `*_reign35_wvk7` fail-closed k=3 · MERGE_DONE `/tmp/r720_merged` 66G/16sh · hard-pin GPUS after mine.env.
