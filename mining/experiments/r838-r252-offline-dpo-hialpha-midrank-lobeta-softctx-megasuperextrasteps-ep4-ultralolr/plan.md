# R838 — Soft Mid Mid Soft MidRank LoBeta UltraLoLR (golden 4,5)

**Axis:** r252 × Offline-DPO × HiAlpha × MidRank × LoBeta × SoftCtx × MegaSuperExtraSteps × ep4 × UltraLoLR
**Base:** `unconst/Affine-5czsc2fc98-r252-merged` @ `b42d6245…`
**Knobs:** β=0.02 · α=128 · r=32 · lr=5e-7 · @12288 · max_steps=19200 · ep=4
**Data:** Soft Mid Mid Soft pairs (same as R828)
**Parent signal:** R825 Short Hi Mid REFUTE ~−0.22× → SoftCtx isolate; R828 Soft Mid Mid Soft MidBeta TRAIN sibling → **LoBeta** isolate; ≠ R828 Midβ / ≠ R802 Soft Mid Lo Soft / ≠ Online / ≠ GRPO
**Decision rule:** Stage-5 iff fresh v4 n80 paired margin > max(2·SE, 0.002) AND median |z|≥80 AND B≥0.30 (re-sim vs reign36 vera if clears on stale tammy)
**GPUs:** golden-comet-78 4,5 · n80 local :8002 after merge
