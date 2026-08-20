# R1018 — MidCtx HiRank Hiβ Mega MidLR

**Status (p4151):** **TRAIN** on `mine-r338` GPUs 4,5 after R1008 CROWN_OK ~1.325×. MERGE→n80 armed.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.3** / 128 / **64** / **1e-6** |
| max_len / steps | **8192** MidCtx / **19200** Mega |
| parent | R1008 MidCtx HiRank Midβ Mega MidLR CROWN_OK m=+0.005917 ~1.325× → Hiβ isolate |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
