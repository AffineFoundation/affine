# R663 — Long HiRank LoBeta Mega ep3×LoLR

**Axis:** r252 Offline-DPO HiAlpha HiRank LoBeta LongCtx MegaExtraSteps epochs=3×LoLR
**Knobs:** β=0.02 · α=128 · r=64 · lr=1e-6 · max_len=16384 · max_steps=3600 · ep=3
**Parent signal:** R648 Long HiRank LoBeta ep2×LoLR ~0.68× (best Long near-miss) / R637 Soft MidRank LoBeta ep3 SIGNAL ~1.45×
**≠** R648 ep2 / ≠ R662 Long MidRank LoBeta ep3 / ≠ R656 Short HiRank LoBeta ep3 / ≠ Online / ≠ GRPO
**Decision:** Stage-5 iff fresh v4 n80 (k=3) paired margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign34
**Pod:** mine-crown-1 GPUs 0,1 (idle after R654 MERGE_DONE)
