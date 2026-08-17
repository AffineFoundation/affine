# R662 — Long MidRank LoBeta Mega ep3×LoLR

**Axis:** r252 Offline-DPO HiAlpha MidRank LoBeta LongCtx MegaExtraSteps epochs=3 lr=1e-6
**Parent:** R643 Long MidRank LoBeta ep2×LoLR ~0.50× (k=1 note) / R579 ~0.46×; sibling of R637 Soft MidRank LoBeta ep3 SIGNAL ~1.45×
**Knobs:** β=0.02 α=128 r=32 lr=1e-6 @16384 max_steps=3600 ep=3
**Pod:** zesty GPUs 4,5 (idle while R634 SCP→chall 6,7; R633 chall later — reap before R633 needs 4,5)
**Decision:** Stage-5 iff fresh v4 n80 (k=3) margin > max(2·SE, 0.002) AND median|z|≥80 AND B≥0.30 vs reign34
**≠** R643 ep2 / R613 @5e-6 / R641 Long MidRank MidBeta / R648 Long HiRank LoBeta / R653 Short MidRank LoBeta / Online / GRPO R583
