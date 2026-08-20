# R1070 — MidCtx MidRank Hiβ Mega HiLR

**Status (p4199):** **TRAIN** on `mine-r339` GPUs 4,5 after R1053 REFUTE reap :8002.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.3** / 128 / **32** / **2e-6** |
| max_len / steps | **8192** / **19200** |
| parent | R1053 MidCtx MidRank Hiβ Ultra HiLR REFUTE m=−0.000766 ~−0.21× thought✓216 B✓0.4125 k=3 → Mega isolate |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
**REFUTE v4 p4199:** m=−0.000814 ~−0.08× thought✓195.5 B✓0.4625 → **R1085**
