# R671 — MidCtx HiRank HiBeta Mega ep3×LoLR

**Axis:** r252 Offline-DPO HiAlpha HiRank HiBeta MidCtx MegaExtraSteps epochs=3 lr=1e-6
**Parent:** R603 MidCtx HiRank HiBeta REFUTE ~0.22×; LoLR ep3 anti-overfit; sibling of R637 Soft MidRank LoBeta ep3 SIGNAL ~1.45×
**Knobs:** β=0.3 α=128 r=64 lr=1e-6 @8192 max_steps=3600 ep=3
**Pod:** brave GPUs 4,5 (idle; R633 SCP uplink live ≠ train)
**Decision:** Stage-5 iff fresh v4 n80 (k=3) margin > max(2·SE, 0.002) AND median|z|≥80 AND B≥0.30 vs reign34
**≠** R603 ep1 / R661 MidCtx HiRank MidBeta ep3 / R655 MidCtx HiRank LoBeta ep3 / R660 MidCtx MidRank HiBeta ep3 / R670 Short HiRank HiBeta / Online / GRPO R583
