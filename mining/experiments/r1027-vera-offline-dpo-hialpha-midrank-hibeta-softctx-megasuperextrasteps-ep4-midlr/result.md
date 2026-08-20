# R1027 — SoftCtx MidRank Hiβ Mega MidLR

**Status (p4161):** **TRAIN** on `mine-r338` GPUs 6,7 after R1017 REFUTE. MERGE→n80 waiter armed.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | 0.3 / 128 / 32 / **1e-6** |
| max_len / steps | 12288 SoftCtx / **19200** Mega |
| parent | R1017 MidCtx MidRank Hiβ Ultra MidLR REFUTE m=+0.000341 ~0.08× |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
