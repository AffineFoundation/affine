# R668 — Long MidRank HiBeta Mega ep3×LoLR

**Axis:** r252 Offline-DPO HiAlpha MidRank HiBeta LongCtx MegaExtraSteps epochs=3 lr=1e-6
**Parent:** R604 Long MidRank HiBeta ~0.26× / R626 ep2@5e-6 inverted hard; anti-overfit LoLR ep3; sibling of R637 Soft MidRank LoBeta ep3 SIGNAL ~1.45×
**Knobs:** β=0.3 α=128 r=32 lr=1e-6 @16384 max_steps=3600 ep=3
**Pod:** zesty GPUs 6,7 (idle after R634 REFUTE; R662 owns 4,5)
**Decision:** Stage-5 iff fresh v4 n80 (k=3) margin > max(2·SE, 0.002) AND median|z|≥80 AND B≥0.30 vs reign34
**≠** R626 ep2@5e-6 / R604 @5e-6 / R662 Long MidRank LoBeta ep3 / R664 Long MidRank MidBeta ep3 / R660 MidCtx MidRank HiBeta / R659 Short MidRank HiBeta / Online / GRPO R583
