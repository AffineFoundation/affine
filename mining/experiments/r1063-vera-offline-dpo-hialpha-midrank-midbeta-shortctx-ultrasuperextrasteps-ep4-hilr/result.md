# R1063 — ShortCtx MidRank Midβ Ultra HiLR

**Status (p4193):** TRAIN armed on `mine-r337` GPUs6,7 after R1048 REFUTE.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.1** / 128 / **32** / **2e-6** |
| ctx / steps | ShortCtx `@6144` / Ultra `28800` |
| parent | R1048 ShortCtx MidRank Midβ Ultra MidLR REFUTE m=+0.000677 ~0.18× → Ultra HiLR isolate |

## Timeline
- p4193: R1048 REFUTE (thought✓199 B✓0.452 k=3) → exact-PID reap :8002 → TRAIN launch

Decision rule: Stage-5 iff margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 (k=3, τ=0.03) vs reign36.
