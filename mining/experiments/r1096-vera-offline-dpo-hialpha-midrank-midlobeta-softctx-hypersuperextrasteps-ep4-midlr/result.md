# R1096 — SoftCtx MidRank MidLoβ HyperExtra MidLR

**Status (p4226):** TRAIN armed on `mine-r340` GPUs1,2 (idle B200 fill; R340 Online still on 6,7).

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.05** / 128 / **32** / **1e-6** |
| ctx / steps | SoftCtx `@12288` / HyperExtra `38400` |
| parent | SoftCtx MidRank MidLoβ Ultra MidLR R1021 REFUTE ~−0.16× → Hyper MidLR isolate; ≠ SoftCtx HiRank MidLoβ Hyper R1090 / ≠ SoftCtx MidRank Midβ Hyper R1083 / ≠ MidCtx MidRank MidLoβ Hyper R1095 / ≠ Online R340 / ≠ GRPO |

Decision rule: Stage-5 iff margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 (k=3, τ=0.03) vs reign36.
