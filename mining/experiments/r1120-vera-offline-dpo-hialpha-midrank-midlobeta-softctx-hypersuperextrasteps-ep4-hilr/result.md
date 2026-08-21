# R1120 — SoftCtx MidRank MidLoβ HyperExtra HiLR

**Status (p4251):** arming TRAIN on `mine-r340` GPUs**1,2** after R1096 REFUTE.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.05** / 128 / **32** / **2e-6** |
| ctx / steps | SoftCtx `@12288` / HyperExtra `38400` |
| parent | SoftCtx MidRank MidLoβ Hyper MidLR **R1096 REFUTE** m=−0.000713 ~−0.19× thought✓157 B✓0.439 → **HiLR isolate** |

Decision rule: Stage-5 iff margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 (k=3, τ=0.03) vs reign36.
