# R851 — tammy Soft Mid Mid Soft × HiRank × β=0.02 × SoftCtx × UltraLoLR
**Axis:** tammyfritz × Offline-DPO × HiAlpha × HiRank × β=0.02 × SoftCtx × MegaSuperExtra × ep4 × UltraLoLR
**Base:** `tammyfritz/Affine-5hmwhnfbix-tammy2` @ `7e5fd5f8…`
**Knobs:** β=0.02 · α=128 · r=64 · lr=5e-7 · @12288 · max_steps=19200 · ep=4
**Data:** Soft Mid Mid Soft pairs (604)
**Parent signal:** R810 Soft Hi Lo Soft HiRank Loβ REFUTE 0.45x → Soft Mid Mid Soft HiRank Loβ SoftCtx UltraLoLR; ≠ Online / ≠ GRPO / ≠ vera R846/R847
**Decision rule:** Stage-5 iff fresh v4 n80 paired margin > max(2·SE, 0.002) AND median |z|≥80 AND B≥0.30 vs **reign36** vera
**GPUs:** brave-raven-a9 6,7 · merge local → host-relay→TKC n80
