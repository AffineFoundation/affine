# R1048 — ShortCtx MidRank Midβ Ultra MidLR

**Status (p4193):** **REFUTE v4** vs reign36.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.1** / 128 / **32** / **1e-6** |
| ctx / steps | ShortCtx `@6144` / Ultra `28800` |
| parent | R1037 SoftCtx Midβ Mega MidLR REFUTE ~−0.79× → ShortCtx isolate |

## Verdict (p4193)
- margin=**+0.000677** SE=0.001912 z=0.354 n=80 bar≈0.003823 (~**0.18×**)
- thought✓(199) B✓(0.452) k=3 τ=0.03 · wins=false
- → exact-PID reap :8002 → **R1063** ShortCtx MidRank Midβ Ultra **HiLR**

## Timeline
- p4179: TRAIN launched GPUs6,7
- 2026-08-20T19:07Z train.done · 19:10Z merge.done
- p4192: waiter path bug fixed; n80 LIVE
- p4193: n80 REFUTE → R1063 TRAIN pid**120682**

Decision rule: Stage-5 iff margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 (k=3, τ=0.03) vs reign36.
