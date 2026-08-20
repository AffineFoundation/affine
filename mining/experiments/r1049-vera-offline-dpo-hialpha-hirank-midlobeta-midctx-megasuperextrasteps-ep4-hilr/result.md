# R1049 — MidCtx HiRank MidLoβ Mega HiLR

Parent: **R1038 REFUTE** m=−0.002795 ~−0.67× (thought✓179.5 B✓0.3375 k=3).

Isolates **HiLR** (lr=`2e-6`) on the MidCtx+HiRank+MidLoβ+Mega lane after UltraLoLR flop.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / r / α | 0.05 / 64 / 128 |
| lr / max_len / steps | **2e-6** / 8192 / 19200 |
| pod | `mine-r338-…` GPUs **4,5** |
| launched | p4180 TRAIN pid**133218** + MERGE→n80 waiter |

Decision rule: Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
