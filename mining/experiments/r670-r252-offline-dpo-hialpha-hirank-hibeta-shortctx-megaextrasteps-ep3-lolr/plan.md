# R670 — Short HiRank HiBeta Mega ep3×LoLR

**Axis:** r252 Offline-DPO HiAlpha HiRank HiBeta ShortCtx MegaExtraSteps epochs=3 lr=1e-6
**Parent:** R606 Short HiRank HiBeta REFUTE ~−0.20× @5e-6; LoLR ep3 anti-overfit; sibling of R637 Soft MidRank LoBeta ep3 SIGNAL ~1.45×
**Knobs:** β=0.3 α=128 r=64 lr=1e-6 @6144 max_steps=3600 ep=3
**Pod:** brave GPUs 2,3 (idle; R633 SCP uplink live ≠ train; R669 on 0,1)
**Decision:** Stage-5 iff fresh v4 n80 (k=3) margin > max(2·SE, 0.002) AND median|z|≥80 AND B≥0.30 vs reign34
**≠** R606 @5e-6 / R666 Short HiRank MidBeta ep3 / R656 Short HiRank LoBeta ep3 / R659 Short MidRank HiBeta ep3 / R667 Soft HiRank HiBeta SoftCtx / Online / GRPO R583
