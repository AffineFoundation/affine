# R1123 — ShortCtx HiRank MidLoβ HyperExtra HiLR

**Status (p4252):** arming TRAIN on `mine-r252` GPUs**6,7** after R1105 REFUTE.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.05** / 128 / **64** / **2e-6** |
| ctx / steps | ShortCtx `@6144` / HyperExtra `38400` |
| parent | ShortCtx HiRank Midβ Hyper HiLR **R1105 REFUTE** m=+0.001862 ~0.51× thought✓179 B✓0.3375 → **MidLoβ isolate** |

Decision rule: Stage-5 iff margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 (k=3, τ=0.03) vs reign36.
