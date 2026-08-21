# R1153 — ShortCtx HiRank MidLoβ HyperExtra UltraLoLR

**Status (p4274):** TRAIN on `mine-r252` GPUs**6,7** after R1123 REFUTE.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.05** / 128 / **64** / **5e-7** |
| ctx / steps | ShortCtx `@6144` / HyperExtra `38400` |
| parent | ShortCtx HiRank MidLoβ Hyper HiLR **R1123 REFUTE** m=−0.000808 ~−0.11× thought✓201 B✓0.447 → **UltraLoLR isolate** |

Decision rule: Stage-5 iff margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 (k=3, τ=0.03) vs reign36.
