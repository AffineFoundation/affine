# R1096 — SoftCtx MidRank MidLoβ HyperExtra MidLR

**Status (p4245):** **CHALL+n80 armed** on `mine-r340` GPUs1,2 :8002 (replaced stub lean; merges were idle since ~02:10Z).

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.05** / 128 / **32** / **1e-6** |
| ctx / steps | SoftCtx `@12288` / HyperExtra `38400` |
| parent | SoftCtx MidRank MidLoβ Ultra MidLR R1021 REFUTE ~−0.16× → Hyper MidLR isolate |
| p4245 | exact-PID reap aborted R340 chall pid24856 (CUDA 4,5 vs king); outer pid**29288** chall pid**29383** |

Decision rule: Stage-5 iff margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 (k=3, τ=0.03) vs reign36.
