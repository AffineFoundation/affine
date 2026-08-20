# R1063 — ShortCtx MidRank Midβ Ultra HiLR

**Status (p4207):** **REFUTE v4** vs reign36.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.1** / 128 / **32** / **2e-6** |
| ctx / steps | ShortCtx `@6144` / Ultra `28800` |
| n80 | m=**−0.000921** SE=0.002868 z=−0.321 n=79 bar≈0.005735 (~**−0.16×**) thought✓(195) B✓(0.430) k=3/τ=0.03 |
| next | **R1078** MidLR restore + HyperExtra 38400 |

## Timeline
- p4193: R1048 REFUTE → TRAIN
- p4207: n80 REFUTE → reap :8002 → R1078 TRAIN

Decision rule: Stage-5 iff margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 (k=3, τ=0.03) vs reign36.
