# R833 — r252 Short Mid Mid Soft UltraLoLR

**Axis:** Offline-DPO HiAlpha MidRank MidBeta ShortCtx MegaSuperExtra ep4 UltraLoLR
**Base:** `unconst/Affine-5czsc2fc98-r252-merged`@`b42d6245`
**Knobs:** β=0.1 α=128 r=32 lr=5e-7 @6144 max_steps=19200 ep=4 GPUs 6,7
**Why:** R819 Short Mid Lo REFUTE ~−1.57× + R816 Short Mid Hi REFUTE ~−0.63× → Midβ isolate
**Decision:** Stage-5 iff n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign35 (v4 k=3)
**Path:** TRAIN → merge → lean chall :8003 → n80
