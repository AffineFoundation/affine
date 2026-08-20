# R1038 — MidCtx HiRank MidLoβ Mega UltraLoLR

Parent: **R1028 REFUTE** m=+0.001248 ~0.23× (thought✓156 B✓0.314 k=3).

Isolates **UltraLoLR** (lr=`5e-7`) on the R1008 MidCtx+HiRank+Mega lane after MidLR flop
(mirrors R959 Mega UltraLoLR crown path).

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / r / α | 0.05 / 64 / 128 |
| lr / max_len / steps | **5e-7** / 8192 / 19200 |
| pod | `mine-r338-…` GPUs **4,5** |
| launched | p4168 TRAIN pid**119904** + MERGE→n80 waiter |

Decision rule: Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
