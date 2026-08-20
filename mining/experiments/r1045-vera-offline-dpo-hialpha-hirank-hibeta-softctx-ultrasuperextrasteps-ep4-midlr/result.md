# R1045 — SoftCtx HiRank Hiβ Ultra MidLR

**Status (p4178):** **TRAIN** on `mine-r924` GPUs 6,7 after R1026 REFUTE. MERGE→n80 waiter armed.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.3** / 128 / **64** / **1e-6** |
| max_len / steps | **12288** / **28800** |
| parent | R1026 SoftCtx HiRank Hiβ Mega MidLR REFUTE m=+0.000483 ~0.08× → Ultra MidLR isolate |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
