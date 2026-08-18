# R730 — Short HiRank MidBeta SuperExtra (r252)

**Axis:** Offline-DPO HiAlpha HiRank MidBeta ShortCtx SuperExtraSteps ep3×LoLR
β=0.1 r=64 α=128 lr=1e-6 @6144 max_steps=**14400**

**Decision rule:** Stage-5 iff fresh v4 n80 `margin > max(2·SE, δ=0.002)` AND median `|z|≥80` AND B pass ≥0.30 vs **reign35** (wvk=7, k=3, τ=0.03).

## Status (p3770)
**TRAIN** on `mine-r262` (golden) GPUs **6,7** pid**544629** + wait→merge armed · kept=604 · after R720 REFUTE · pin BASE after mine.env.
