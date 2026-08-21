# R1167 — ShortCtx HiRank MidLoβ HyperExtra MidLR

**Status (p4287):** TRAIN on `mine-r252` GPUs**6,7** after R1153 REFUTE.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.05** / 128 / **64** / **1e-6** |
| ctx / steps | ShortCtx `@6144` / HyperExtra `38400` |
| parent | ShortCtx HiRank MidLoβ Hyper UltraLoLR **R1153 REFUTE** m=+0.002569 SE=0.002314 ~**0.56×** thought✓155 B✓0.397 → **MidLR isolate** |

Decision rule: Stage-5 iff margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 (k=3, τ=0.03) vs reign36.
≠ UltraLoLR R1153 / ≠ HiLR R1123 / ≠ SoftCtx HiRank MidLoβ UltraLoLR R1138 / ≠ MidCtx HiRank MidLoβ UltraLoLR R1141 / ≠ ShortCtx MidRank MidLoβ UltraLoLR R1161 / ≠ Online / ≠ GRPO
