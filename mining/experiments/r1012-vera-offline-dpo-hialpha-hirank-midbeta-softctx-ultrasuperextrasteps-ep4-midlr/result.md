# R1012 — SoftCtx HiRank Midβ Ultra MidLR

**Status (p4145):** **TRAIN** on `mine-r938` GPUs 2,3 after R992 REFUTE. MERGE→n80 waiter armed.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | 0.1 / 128 / 64 / **1e-6** |
| max_len / steps | 12288 SoftCtx / **28800** Ultra |
| parent | R992 Mega MidLR REFUTE m=+0.001984 ~0.50× |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
