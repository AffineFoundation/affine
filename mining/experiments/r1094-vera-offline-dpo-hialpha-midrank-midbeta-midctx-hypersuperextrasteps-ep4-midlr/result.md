# R1094 — MidCtx MidRank Midβ HyperExtra MidLR

**Status (p4224):** TRAIN armed on `mine-r337` GPUs4,5 after R1064 idle chall reap (QUEUED chal-00974).

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.1** / 128 / **32** / **1e-6** |
| ctx / steps | MidCtx `@8192` / HyperExtra `38400` |
| parent | R1064 MidCtx MidRank Midβ Ultra HiLR CROWN_OK ~1.021× → Hyper MidLR isolate |

## Timeline
- p4224: R1064 still QUEUED chal-00974; idle :8003 chall reaped → TRAIN launch
- B300/B200×8 stock=0 this pass; waiters stay armed

Decision rule: Stage-5 iff margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 (k=3, τ=0.03) vs reign36.
