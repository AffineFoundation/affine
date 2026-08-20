# R1016 — MidCtx HiRank Midβ Ultra MidLR

**Status (p4150):** **TRAIN** on `mine-r924` GPUs 4,5 after R1003 REFUTE. MERGE→n80 waiter armed.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.1** / 128 / **64** / **1e-6** |
| max_len / steps | **8192** MidCtx / **28800** Ultra |
| parent | R1003 Ultra UltraLoLR REFUTE m=+0.002668 ~0.40× |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
