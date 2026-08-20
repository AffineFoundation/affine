# R1042 — ShortCtx MidRank MidLoβ Mega MidLR

**Status (p4174):** **TRAIN** on `mine-crown-1` GPUs 6,7 after parent REFUTE. MERGE→n80 waiter armed.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.05** / 128 / **32** / **1e-6** |
| max_len / steps | **6144** / **19200** |
| parent | R1035 ShortCtx MidRank MidLoβ Ultra MidLR REFUTE m=-0.006514 ~-0.63× thought✓180 B✓0.392 k=3 → Mega MidLR isolate; ≠ Ult… |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
