# R676 — Long HiRank LoBeta LongCtx UltraExtra ep3×LoLR

**Axis:** r252 Offline-DPO HiAlpha HiRank LoBeta LongCtx UltraExtraSteps ep3 LoLR
**Knobs:** β=0.02 α=128 r=64 lr=1e-6 @16384 max_steps=7200 ep=3
**Parent:** amplify R663 Mega ep3 MERGE_DONE / R648 ~0.68× best Long near-miss with 2× steps
**Distinct from:** R663 Mega 3600; R674 Long MidRank LoBeta UltraExtra; R668 Long MidRank HiBeta; R665 Long HiRank MidBeta; R669 Long HiRank HiBeta
**Pod:** mine-crown-1 GPUs 0,1 (post R663 MERGE; R653 n80 + R655 SCP live elsewhere)
**Decision:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign34
