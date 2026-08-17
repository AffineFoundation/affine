# R661 — MidCtx HiRank MidBeta Mega ep3×LoLR

**Axis:** r252 Offline-DPO HiAlpha HiRank MidBeta MidCtx MegaExtraSteps epochs=3 lr=1e-6
**Parent:** R644 MidCtx HiRank MidBeta ep2×LoLR ~−0.07× REFUTE; R597 ~0.49×; R612 @5e-6 weak; sibling of R637 Soft MidRank LoBeta ep3 SIGNAL ~1.45×
**Knobs:** β=0.1 α=128 r=64 lr=1e-6 @8192 max_steps=3600 ep=3
**Pod:** brave GPUs 6,7 (idle while R653 0,1 / R659 2,3 / R660 4,5 / R634 SCP live; train ≠ dual-pipe)
**Decision:** Stage-5 iff fresh v4 n80 (k=3) margin > max(2·SE, 0.002) AND median|z|≥80 AND B≥0.30 vs reign34
**≠** R644 ep2 / R612 @5e-6 / R655 MidCtx HiRank LoBeta / R657 MidCtx MidRank MidBeta / R654 MidCtx MidRank LoBeta / R660 MidCtx MidRank HiBeta / Online / GRPO R583
