# R1032 — ShortCtx MidRank Hiβ Ultra MidLR

| field | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| method | Offline DPO (Reason-pair prefs) |
| β / α / r / lr | **0.3** / 128 / 32 / **1e-6** |
| max_len / steps | **6144** ShortCtx / **28800** Ultra |
| parent | R1015 ShortCtx MidRank Hiβ Mega MidLR REFUTE m=−0.002353 ~−0.94× |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.

**Isolate:** Mega→Ultra on R1015 lane. ≠ Mega MidLR R1015 / ≠ Mega UltraLoLR R1001 / ≠ Ultra HiLR R984 / ≠ SoftCtx MidRank Hiβ Mega MidLR R1027 / ≠ MidCtx MidRank Hiβ Ultra MidLR R1017 / ≠ Online / ≠ GRPO.
