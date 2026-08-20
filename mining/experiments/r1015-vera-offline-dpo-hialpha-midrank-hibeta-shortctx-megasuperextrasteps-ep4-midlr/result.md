# R1015 — ShortCtx MidRank Hiβ Mega MidLR

**Status (p4149):** **TRAIN** on `mine-r924` GPUs 1,3 after R1001 REFUTE. MERGE→n80 waiter armed.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.3** / 128 / 32 / **1e-6** |
| max_len / steps | **6144** ShortCtx / **19200** Mega |
| parent | R1001 Mega UltraLoLR REFUTE m=−0.001264 ~−0.36× |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
