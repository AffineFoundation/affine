# R660 — MidCtx MidRank HiBeta Mega ep3×LoLR

**Axis:** r252 Offline-DPO HiAlpha MidRank HiBeta MidCtx MegaExtraSteps epochs=3 lr=1e-6
**Parent:** R650 MidCtx MidRank HiBeta ep2×LoLR ~0.47× REFUTE; R625 @5e-6 ~0.09×; sibling of R637 Soft MidRank LoBeta ep3 SIGNAL ~1.45×
**Knobs:** β=0.3 α=128 r=32 lr=1e-6 @8192 max_steps=3600 ep=3
**Pod:** brave GPUs 4,5 (idle while R653 0,1 / R659 2,3 / R634 SCP live; train ≠ dual-pipe)
**Decision:** Stage-5 iff fresh v4 n80 (k=3) margin > max(2·SE, 0.002) AND median|z|≥80 AND B≥0.30 vs reign34
**≠** R650 ep2 / R625 @5e-6 / R654 MidCtx MidRank LoBeta / R657 MidCtx MidRank MidBeta / R659 Short MidRank HiBeta / R655 MidCtx HiRank LoBeta / Online / GRPO R583
