# R1083 — SoftCtx MidRank Midβ HyperExtra MidLR

**Status (p4217):** TRAIN armed on `mine-r337` GPUs6,7 after R1078 REFUTE.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.1** / 128 / **32** / **1e-6** |
| ctx / steps | SoftCtx `@12288` / HyperExtra `38400` |
| parent | R1078 ShortCtx MidRank Midβ Hyper MidLR REFUTE m=−0.004523 ~−0.71× → SoftCtx isolate |

## Timeline
- p4217: R1078 REFUTE (thought✓185 B✓0.481 k=3) → exact-PID reap :8002 → TRAIN launch
- B300/B200×8 stock=0 this pass (ghost B200 cleared); waiters stay armed

Decision rule: Stage-5 iff margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 (k=3, τ=0.03) vs reign36.
