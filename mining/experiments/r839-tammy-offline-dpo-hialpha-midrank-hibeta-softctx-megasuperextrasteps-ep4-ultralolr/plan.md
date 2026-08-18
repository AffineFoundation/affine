# R839 — Soft Mid Mid Soft MidRank HiBeta SoftCtx UltraLoLR (crown 4,5)

**Axis:** tammy × Offline-DPO × HiAlpha × MidRank × HiBeta × SoftCtx × MegaSuperExtraSteps × ep4 × UltraLoLR
**Base:** `tammyfritz/Affine-5hmwhnfbix-tammy2` @ `7e5fd5f8…`
**Knobs:** β=0.3 · α=128 · r=32 · lr=5e-7 · @12288 · max_steps=19200 · ep=4
**Data:** Soft Mid Mid Soft pairs (from R829)
**Parent signal:** R829 Soft Mid Mid Soft LoBeta ~0.35× REFUTE → Soft Mid Mid Soft HiBeta isolate; ≠ R829 Loβ / ≠ R800 Soft Mid Hi Soft / ≠ R827 MidCtx Hi Mid / ≠ Online / ≠ GRPO
**Decision rule:** Stage-5 iff fresh v4 n80 paired margin > max(2·SE, 0.002) AND median |z|≥80 AND B≥0.30 vs **reign36** vera
**GPUs:** crown gentle-orbit-bd 4,5 · n80 local :8002 after merge
