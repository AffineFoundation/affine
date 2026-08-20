# R1065 — ShortCtx MidRank Hiβ Ultra HiLR

Parent: **R1050 REFUTE** m=−0.003658 ~−0.44× (thought✓189 B✓0.530 k=3).

Isolates **ShortCtx** (`max_len=6144`) on the R1032 crown lane (ShortCtx MidRank Hiβ Ultra) with **HiLR** (`2e-6`). SoftCtx HiLR flopped; MidLR already crowned as R1032.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / r / α | 0.3 / 32 / 128 |
| lr / max_len / steps | **2e-6** / **6144** / 28800 |
| pod | `mine-r338-…` GPUs **6,7** :8002 |

## Timeline
- p4196: R1050 REFUTE → exact-PID reap :8002 → TRAIN + MERGE→n80 waiter

Decision rule: Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
