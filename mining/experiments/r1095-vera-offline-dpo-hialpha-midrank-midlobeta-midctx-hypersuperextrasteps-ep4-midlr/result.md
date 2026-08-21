# R1095 — MidCtx MidRank MidLoβ HyperExtra MidLR

**Status (p4225):** TRAIN armed on `mine-r924` GPUs4,5 after R1068 idle chall reap.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.05** / 128 / **32** / **1e-6** |
| ctx / steps | MidCtx `@8192` / HyperExtra `38400` |
| parent | R1047 MidCtx MidRank MidLoβ Ultra HiLR REFUTE ~0.19× → Hyper MidLR isolate; R1068 idle :8004 reaped |

## Timeline
- p4225: R1068 REFUTE idle on r924 :8004 → TRAIN launch; crown SSH timeout (brave-comet-f4); B300/B200×8 stock=0

Decision rule: Stage-5 iff margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 (k=3, τ=0.03) vs reign36.
