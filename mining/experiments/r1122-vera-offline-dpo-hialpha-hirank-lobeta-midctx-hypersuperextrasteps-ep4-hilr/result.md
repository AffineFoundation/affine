# R1122 — MidCtx HiRank Loβ HyperExtra HiLR

**Status (p4251):** arming TRAIN on `mine-r338` GPUs**4,5** after R1111 REFUTE.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.02** / 128 / **64** / **2e-6** |
| ctx / steps | MidCtx `@8192` / HyperExtra `38400` |
| parent | MidCtx HiRank Hiβ Hyper HiLR **R1111 REFUTE** m=+0.003700 ~0.68× thought✓199 B✓0.515 → **Loβ isolate** (SoftCtx HiRank Hiβ Hyper HiLR R1114 + ShortCtx R1109 already TRAIN) |

Decision rule: Stage-5 iff margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 (k=3, τ=0.03) vs reign36.
