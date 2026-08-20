# R1052 — ShortCtx HiRank Hiβ Ultra MidLR

Parent: **R1032 CROWN_OK** m=+0.005461 ~1.049× (QUEUED chal-00967). Isolates **HiRank** (r=`64`) on the ShortCtx+Hiβ+Ultra+MidLR lane that cleared the live crown rule under MidRank.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / r / α | 0.3 / **64** / 128 |
| lr / max_len / steps | 1e-6 / **6144** / 28800 |
| pod | `mine-r339-…` GPUs **6,7** · chall :**8003** (R339 n80 keeps :8002) |
| launched | p4182 TRAIN + MERGE→n80 waiter |

≠ MidRank R1032 CROWN / ≠ ShortCtx MidRank Midβ Ultra MidLR R1048 / ≠ ShortCtx MidRank MidLoβ Ultra R1035 / ≠ SoftCtx HiRank Hiβ Ultra MidLR R1045 / ≠ Online R339 / ≠ GRPO

Decision rule: Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
