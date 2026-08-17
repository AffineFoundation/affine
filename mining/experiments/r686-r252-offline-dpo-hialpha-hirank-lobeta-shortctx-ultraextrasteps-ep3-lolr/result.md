# R686 result

**Status (p3712):** TRAIN launched on zesty-comet-da GPUs **4,5** pid**803242** @ 2026-08-17T18:46:51Z; wait→merge armed.

**Axis:** r252 Offline-DPO HiAlpha HiRank LoBeta ShortCtx UltraExtraSteps ep3×LoLR  
β=0.02 α=128 r=64 lr=1e-6 @6144 max_steps=7200 ep=3 · kept=604

**Parent:** amplify R656 Mega / R617 ~0.64× with 2× steps (≠ Mega 3600 / ≠ R683 Short MidRank HiBeta / ≠ R679 Short HiRank HiBeta / ≠ R676 Long HiRank LoBeta).

**Decision rule:** Stage-5 iff fresh v4 (k=3 τ=0.03) n80 paired margin > max(2·SE, 0.002) AND median |z|≥80 AND B pass≥0.30 vs reign34.
