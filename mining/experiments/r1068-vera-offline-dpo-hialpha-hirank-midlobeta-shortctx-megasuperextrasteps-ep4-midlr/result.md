# R1068 — ShortCtx HiRank MidLoβ Mega MidLR

**Status (p4198→p4219):** **REFUTE v4** vs reign36.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.05** / 128 / **64** / **1e-6** |
| max_len / steps | **6144** / **19200** |
| n80 | m=**+0.000268** SE=0.001642 z=0.163 n=80 bar≈0.003284 (~**0.08×**) thought✓165 B✓0.364 k=3 |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
