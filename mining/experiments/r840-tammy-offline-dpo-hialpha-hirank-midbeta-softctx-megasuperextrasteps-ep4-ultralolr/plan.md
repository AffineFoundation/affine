# R840 — Soft Mid Mid Soft HiRank MidBeta SoftCtx UltraLoLR (crown 6,7)

**Axis:** tammy × Offline-DPO × HiAlpha × HiRank × MidBeta × SoftCtx × MegaSuperExtraSteps × ep4 × UltraLoLR
**Base:** `tammyfritz/Affine-5hmwhnfbix-tammy2` @ `7e5fd5f8…`
**Knobs:** β=0.1 · α=128 · r=64 · lr=5e-7 · @12288 · max_steps=19200 · ep=4
**Data:** Soft Mid Mid Soft pairs (from R829)
**Parent signal:** R829 Soft Mid Mid Soft LoBeta ~0.35× + R795 Soft Hi Mid Soft near-parity → Soft Mid Mid Soft HiRank MidBeta SoftCtx UltraLoLR; ≠ R829 MidRank Loβ / ≠ R788 Soft Hi Mid Soft / ≠ Online / ≠ GRPO
**Decision rule:** Stage-5 iff fresh v4 n80 paired margin > max(2·SE, 0.002) AND median |z|≥80 AND B≥0.30 vs **reign36** vera
**GPUs:** crown gentle-orbit-bd 6,7 · n80 local :8003 after merge
