# R1112 — MidCtx MidRank MidLoβ HyperExtra HiLR

**Status (p4241):** TRAIN armed on `mine-r924` GPUs4,5 after R1095 REFUTE chall reap.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.05** / 128 / **32** / **2e-6** |
| ctx / steps | MidCtx `@8192` / HyperExtra `38400` |
| parent | R1095 MidCtx MidRank MidLoβ Hyper MidLR REFUTE m=−0.001309 ~−0.53× → HiLR isolate |

Decision rule: Stage-5 iff margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 (k=3, τ=0.03) vs reign36.
