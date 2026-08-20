# R1061 — MidCtx HiRank Midβ Mega HiLR

**Status (p4190):** **TRAIN** on `mine-r338` GPUs 4,5 after R1049 REFUTE reap :8003.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.1** / 128 / **64** / **2e-6** |
| max_len / steps | **8192** / **19200** |
| parent | R1049 MidCtx HiRank MidLoβ Mega HiLR REFUTE m=−0.001495 ~−0.16× thought✓214 B✓0.557 k=3 → Midβ isolate |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
