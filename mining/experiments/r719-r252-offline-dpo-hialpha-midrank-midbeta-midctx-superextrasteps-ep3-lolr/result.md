# R719 — MidCtx MidRank MidBeta SuperExtra (r252)

**Axis:** Offline-DPO HiAlpha MidRank MidBeta MidCtx SuperExtraSteps ep3×LoLR
β=0.1 r=32 α=128 lr=1e-6 @8192 max_steps=**14400**

**Parent signal:** amplify R698 MidCtx MidRank MidBeta HyperExtra REFUTE
~−0.38× / R682 UltraExtra / R675 Soft MidRank MidBeta ~0.97× with SuperExtra.
≠ HyperExtra 10800 R698 / ≠ UltraExtra 7200 R682 / ≠ Soft MidRank MidBeta
SoftCtx SuperExtra R710 / ≠ Short MidRank MidBeta SuperExtra R720.

**Decision rule (pre-registered):** Stage-5 iff fresh v4 n80
`margin > max(2·SE, δ=0.002)` AND median `|z|≥80` AND B pass ≥0.30
vs reign34 (wvk=7, k=3, τ=0.03).

## Status (p3755)
**TRAIN** on `mine-r262-kevin-v5-nonking-grpo-1` (golden) GPUs **4,5** after
reaping leftover R683 chall :8003; pid**526570**; wait→merge armed.
Keep `/tmp/r683_merged`.
