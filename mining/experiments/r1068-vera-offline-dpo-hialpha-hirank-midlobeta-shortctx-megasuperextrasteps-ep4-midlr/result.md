# R1068 — ShortCtx HiRank MidLoβ Mega MidLR

**Status (p4198):** **TRAIN** on `mine-r924` GPUs 4,5 after R1046 REFUTE reap :8004.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.05** / 128 / **64** / **1e-6** |
| max_len / steps | **6144** / **19200** |
| parent | R1046 MidCtx HiRank MidLoβ Mega MidLR REFUTE m=−0.000756 ~−0.22× thought✓185 B✓0.40 k=3 → ShortCtx isolate |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
