# R1035 — ShortCtx MidRank MidLoβ Ultra MidLR

**Status (p4166):** **TRAIN** on `mine-crown-1` GPUs 6,7 after R1021 REFUTE. MERGE→n80 waiter armed. pid**196551**.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.05** / 128 / **32** / **1e-6** |
| max_len / steps | **6144** ShortCtx / **28800** Ultra |
| parent | R1021 SoftCtx MidRank MidLoβ Ultra MidLR REFUTE ~−0.16× |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
