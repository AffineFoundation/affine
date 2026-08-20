# R1053 — MidCtx MidRank Hiβ Ultra HiLR

Parent: **R339 REFUTE** m=+0.001071 ~0.48× (Online-DPO) + **R1017** MidCtx MidRank Hiβ Ultra MidLR ~0.08× → **HiLR** isolate.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / r / α | 0.3 / 32 / 128 |
| lr / max_len / steps | **2e-6** / **8192** / 28800 |
| pod | `mine-r339` GPUs **4,5** · chall :**8002** |
| launched | p4182 after R339 REFUTE reap |

Decision rule: Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
