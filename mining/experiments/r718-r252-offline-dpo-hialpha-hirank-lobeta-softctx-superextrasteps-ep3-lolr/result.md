# R718 — Soft HiRank LoBeta SoftCtx SuperExtra (r252)

**Axis:** Offline-DPO HiAlpha HiRank LoBeta SoftCtx SuperExtraSteps ep3×LoLR
β=0.02 r=64 α=128 lr=1e-6 @12288 max_steps=**14400**

**Parent signal:** amplify R691 Soft HiRank LoBeta SoftCtx UltraExtra REFUTE
~−0.44× / R637 Soft MidRank LoBeta SIGNAL ~1.45× with SuperExtra.
≠ UltraExtra 7200 R691 / ≠ MidRank LoBeta SuperExtra R717 / ≠ Soft HiRank
MidBeta SuperExtra R715 / ≠ Soft HiRank HiBeta SuperExtra R716.

**Decision rule (pre-registered):** Stage-5 iff fresh v4 n80
`margin > max(2·SE, δ=0.002)` AND median `|z|≥80` AND B pass ≥0.30
vs reign34 (wvk=7, k=3, τ=0.03).

## Status (p3754)
**TRAIN** on `mine-crown-1` GPUs **6,7** after R696 REFUTE; pid**98310**;
wait→merge armed. Keep `/tmp/r696_merged` / `/tmp/r693_merged`.
