# R669 — Long HiRank HiBeta Mega ep3×LoLR

**Axis:** r252 Offline-DPO HiAlpha HiRank HiBeta LongCtx MegaExtraSteps epochs=3 lr=1e-6
**Parent:** R605 Long HiRank HiBeta ~−0.26× (vs r252); anti-overfit LoLR ep3; sibling of R637 Soft MidRank LoBeta ep3 SIGNAL ~1.45×
**Knobs:** β=0.3 α=128 r=64 lr=1e-6 @16384 max_steps=3600 ep=3
**Pod:** brave GPUs 0,1 (idle after R653 MERGE_DONE; R659/R660/R661 on 2–7; R651 SCP uplink live ≠ train)
**Decision:** Stage-5 iff fresh v4 n80 (k=3) margin > max(2·SE, 0.002) AND median|z|≥80 AND B≥0.30 vs reign34
**≠** R605 @5e-6 / R663 Long HiRank LoBeta ep3 / R665 Long HiRank MidBeta ep3 / R668 Long MidRank HiBeta ep3 / R667 Soft HiRank HiBeta SoftCtx / Online / GRPO R583
