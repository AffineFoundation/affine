# R681 — MidCtx HiRank HiBeta UltraExtra ep3×LoLR

**Axis:** r252 Offline-DPO HiAlpha HiRank HiBeta MidCtx UltraExtraSteps epochs=3 lr=1e-6
**Parent:** R671 MidCtx HiRank HiBeta Mega MERGE_DONE / R603 ~0.22× — UltraExtra steps=7200 (2× Mega)
**Knobs:** β=0.3 α=128 r=64 lr=1e-6 @8192 max_steps=7200 ep=3
**Pod:** brave GPUs 4,5 (idle after R671 MERGE; R663/R655 SCP ≠ this uplink)
**Decision:** Stage-5 iff fresh v4 n80 (k=3) margin > max(2·SE, 0.002) AND median|z|≥80 AND B≥0.30 vs reign34
**≠** R671 Mega 3600 / R679 Short HiRank HiBeta UltraExtra / R661 MidCtx HiRank MidBeta / R655 MidCtx HiRank LoBeta / R660 MidCtx MidRank HiBeta / Online / GRPO R583
