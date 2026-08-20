# R1069 — MidCtx MidRank Loβ Ultra HiLR

**Status (p4199):** **TRAIN** on `mine-crown-1` GPUs 4,5 after R1059 REFUTE reap :8003.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.02** / 128 / **32** / **2e-6** |
| max_len / steps | **8192** / **28800** |
| parent | R1059 ShortCtx MidRank Loβ Ultra HiLR REFUTE m=−0.000889 ~−0.12× thought✓202.5 B✓0.5125 k=3 → MidCtx isolate |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
