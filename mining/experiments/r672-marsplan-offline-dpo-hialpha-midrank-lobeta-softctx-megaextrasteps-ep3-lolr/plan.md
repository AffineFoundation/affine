# R672 — Marsplan Soft MidRank LoBeta Mega ep3×LoLR

**Axis:** marsplan0624 Offline-DPO HiAlpha MidRank LoBeta SoftCtx MegaExtraSteps epochs=3 lr=1e-6 (non-king base)
**Parent:** R637 Soft MidRank LoBeta ep3 SIGNAL ~1.45× on r252 — transfer winning recipe onto `marsplan0624/affine-5gedzafcvg-queen`
**Knobs:** β=0.02 α=128 r=32 lr=1e-6 @12288 max_steps=3600 ep=3
**Pod:** brave GPUs 6,7 (idle; R633 SCP uplink live ≠ train)
**Decision:** Stage-5 iff fresh v4 n80 (k=3) margin > max(2·SE, 0.002) AND median|z|≥80 AND B≥0.30 vs reign34
**≠** R637 r252 Soft MidRank LoBeta / R498 marsplan Soft MidRank LoBeta @5e-6 / R669 Long HiRank HiBeta / Online / GRPO R583
