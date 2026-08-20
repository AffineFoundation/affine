# R1059 — ShortCtx MidRank Loβ Ultra HiLR

**Status (p4188):** **TRAIN** on `mine-crown-1` GPUs 4,5 after R1043 REFUTE reap :8003.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.02** / 128 / **32** / **2e-6** |
| max_len / steps | **6144** / **28800** |
| parent | R1043 SoftCtx MidRank Loβ Ultra HiLR REFUTE m=−0.015244 ~−1.21× thought✓201 B✓0.4625 k=3 → ShortCtx isolate |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
