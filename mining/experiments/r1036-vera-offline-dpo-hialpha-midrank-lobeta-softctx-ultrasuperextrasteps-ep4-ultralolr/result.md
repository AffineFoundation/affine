# R1036 — SoftCtx MidRank Loβ Ultra UltraLoLR

**Status (p4166):** **TRAIN** on `mine-crown-1` GPUs 4,5 after R1022 REFUTE. MERGE→n80 waiter armed. pid**196554**.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.02** / 128 / **32** / **5e-7** |
| max_len / steps | **12288** SoftCtx / **28800** Ultra |
| parent | R1022 SoftCtx MidRank Loβ Ultra MidLR REFUTE ~−0.56× |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
