# R1124 — ShortCtx MidRank Hiβ HyperExtra HiLR

**Status (p4252):** arming TRAIN on `mine-r338` GPUs**6,7** after R1102 REFUTE.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.3** / 128 / **32** / **2e-6** |
| ctx / steps | ShortCtx `@6144` / HyperExtra `38400` |
| parent | SoftCtx MidRank Hiβ Hyper HiLR **R1102 REFUTE** m=−0.004769 ~−0.56× thought✓189 B✓0.4875 → **ShortCtx isolate** |

Decision rule: Stage-5 iff margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 (k=3, τ=0.03) vs reign36.
