# R1019 — MidCtx LoRank Midβ Mega MidLR (B-fix)

**Status (p4153):** **TRAIN** on `mine-crown-1` GPUs 1,3 after R1006 REFUTE (causality_fail). MERGE→n80 waiter armed.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.1** / 128 / **16** / **1e-6** |
| max_len / steps | **8192** MidCtx / **19200** Mega |
| parent | R1006 SoftCtx LoRank Midβ Mega MidLR REFUTE m=+0.001417 ~0.71× B✗0.2875 |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
