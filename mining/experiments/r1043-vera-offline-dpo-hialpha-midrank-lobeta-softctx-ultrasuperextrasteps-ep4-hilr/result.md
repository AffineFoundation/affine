# R1043 — SoftCtx MidRank Loβ Ultra HiLR

**Status (p4174):** **TRAIN** on `mine-crown-1` GPUs 4,5 after parent REFUTE. MERGE→n80 waiter armed.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.02** / 128 / **32** / **2e-6** |
| max_len / steps | **12288** / **28800** |
| parent | R1036 SoftCtx MidRank Loβ Ultra UltraLoLR REFUTE m=+0.000315 ~0.15× thought✓164 B✓0.425 k=3 → Ultra HiLR isolate; ≠ Ultr… |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
