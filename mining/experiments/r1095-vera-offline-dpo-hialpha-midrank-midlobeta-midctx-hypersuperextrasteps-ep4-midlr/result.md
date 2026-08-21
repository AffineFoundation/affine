# R1095 — MidCtx MidRank MidLoβ HyperExtra MidLR

**Status (p4241):** **REFUTE v4** vs reign36.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.05** / 128 / **32** / **1e-6** |
| ctx / steps | MidCtx `@8192` / HyperExtra `38400` |
| n80 | m=**−0.001309** SE=0.001241 z=−1.054 n=78 bar≈0.002483 (~**−0.53×**) thought✓(165) B✓(0.526) k=3/τ=0.03 |

## Timeline
- p4225: R1068 REFUTE idle on r924 :8004 → TRAIN launch
- p4241: n80 DONE → **REFUTE** → reap :8004 → **R1112** Hyper HiLR isolate

Decision rule: Stage-5 iff margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 (k=3, τ=0.03) vs reign36.
