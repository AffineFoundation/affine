# R769 — r252 Short MidRank HiBeta MegaSuperExtra ep4×LoLR

## Axis
- base: `unconst/Affine-5czsc2fc98-r252-merged`@`b42d6245…`
- Offline-DPO HiAlpha MidRank HiBeta ShortCtx MegaSuperExtra ep4 LoLR
- β=0.3 α=128 r=32 lr=1e-6 @6144 max_steps=19200 epochs=4

## Why
R759 Short HiRank HiBeta Mega REFUTE m=−0.001882 ~−0.47× (thought✓160.5 B✓0.385).
Amplify R725/R735 Short MidRank HiBeta near-parity + R683 Ultra WIN lineage with MegaSuperExtra ep4.
Distinct from R760 Short MidRank MidBeta Mega (TRAIN 6,7) and Soft MidRank HiBeta Mega R762.

## Decision rule
Stage-5 iff fresh v4 n80 paired margin > max(2·SE, δ=0.002) AND median |z|≥80 AND B≥0.30 vs reign35.
