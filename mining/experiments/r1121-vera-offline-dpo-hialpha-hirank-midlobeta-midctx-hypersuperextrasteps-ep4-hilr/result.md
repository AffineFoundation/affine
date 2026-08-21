# R1121 — MidCtx HiRank MidLoβ HyperExtra HiLR

**Status (p4251):** arming TRAIN on `mine-r340` GPUs**3,4** after R1097 REFUTE.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.05** / 128 / **64** / **2e-6** |
| ctx / steps | MidCtx `@8192` / HyperExtra `38400` |
| parent | MidCtx HiRank MidLoβ Hyper MidLR **R1097 REFUTE** m=−0.002077 ~−0.45× thought✓148 B✓0.432 → **HiLR isolate** |

Decision rule: Stage-5 iff margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 (k=3, τ=0.03) vs reign36.
