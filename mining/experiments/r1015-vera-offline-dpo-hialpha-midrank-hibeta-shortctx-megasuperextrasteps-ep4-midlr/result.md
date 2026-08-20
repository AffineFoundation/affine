# R1015 — ShortCtx MidRank Hiβ Mega MidLR

| field | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| method | Offline DPO (Reason-pair prefs) |
| β / α / r / lr | **0.3** / 128 / 32 / **1e-6** |
| max_len / steps | **6144** ShortCtx / **19200** Mega |
| parent | R1001 Mega UltraLoLR REFUTE m=−0.001264 ~−0.36× |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.

## Result (p4165 harvest)
**REFUTE v4** vs reign36 · m=**−0.002353** SE=0.001257 z=−1.873 n=80 bar≈0.002514 (~**−0.94×**) thought✓(167) B✓(0.403) k=3/τ=0.03 → **R1032** Ultra MidLR isolate.
