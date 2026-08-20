# R1041 — MidCtx LoRank Midβ Mega HiLR

**Status (p4174):** **TRAIN** on `mine-crown-1` GPUs 1,3 after parent REFUTE. MERGE→n80 waiter armed.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.1** / 128 / **16** / **2e-6** |
| max_len / steps | **8192** / **19200** |
| parent | R1031 MidCtx LoRank Midβ Ultra MidLR REFUTE m=+0.001897 ~0.32× thought✓194 B✓0.470 k=3 → Mega HiLR isolate (R1019 Mega M… |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
