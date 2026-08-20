# R1078 — ShortCtx MidRank Midβ HyperExtra MidLR

**Status (p4207):** TRAIN armed on `mine-r337` GPUs6,7 after R1063 REFUTE.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.1** / 128 / **32** / **1e-6** |
| ctx / steps | ShortCtx `@6144` / HyperExtra `38400` |
| parent | R1063 ShortCtx MidRank Midβ Ultra HiLR REFUTE m=−0.000921 ~−0.16× (R1048 MidLR was +0.18×) → MidLR restore + HyperExtra |

## Timeline
- p4207: R1063 REFUTE (thought✓195 B✓0.430 k=3) → exact-PID reap :8002 → TRAIN launch
- Also p4207: SN120 α→TAO→Lium (59α / τ3.36 → Lium +~$745)

Decision rule: Stage-5 iff margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 (k=3, τ=0.03) vs reign36.
