# R720 — Short MidRank MidBeta SuperExtra (r252)

**Axis:** Offline-DPO HiAlpha MidRank MidBeta ShortCtx SuperExtraSteps ep3×LoLR
β=0.1 r=32 α=128 lr=1e-6 @6144 max_steps=**14400**

**Parent signal:** amplify R688 Short MidRank MidBeta UltraExtra MERGE / R658
Mega / R675 Soft MidRank MidBeta ~0.97× / R683 Short MidRank HiBeta LOST with
SuperExtra. ≠ UltraExtra 7200 R688 / ≠ Soft MidRank MidBeta SoftCtx SuperExtra
R710 / ≠ MidCtx MidRank MidBeta SuperExtra R719 / ≠ R696 Short MidRank LoBeta
UltraExtra REFUTE / ≠ R683 Short MidRank HiBeta UltraExtra LOST.

**Decision rule (pre-registered):** Stage-5 iff fresh v4 n80
`margin > max(2·SE, δ=0.002)` AND median `|z|≥80` AND B pass ≥0.30
vs reign34 (wvk=7, k=3, τ=0.03).

## Status (p3755)
**TRAIN** on `mine-r262-kevin-v5-nonking-grpo-1` (golden) GPUs **6,7** after
reaping leftover R637 chall :8004; pid**526573**; wait→merge armed.
Keep `/tmp/r683_merged` / `/tmp/r637_merged`.
