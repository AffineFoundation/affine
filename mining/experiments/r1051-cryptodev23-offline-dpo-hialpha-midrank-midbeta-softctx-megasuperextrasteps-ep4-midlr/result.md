# R1051 — cryptoDev SoftCtx MidRank Midβ Mega MidLR

Parent: **R1025 REFUTE** m=+0.000156 ~0.028× (thought✓224 B✓0.375 k=3).

Isolates **Mega** (steps=`19200`) vs Ultra MidLR flop on cryptoDev SoftCtx MidRank Midβ.

| knob | value |
|---|---|
| base | `cryptoDev23/Affine-5Dku3dYp9j-hk8161`@`55b7ffe0` |
| β / r / α | 0.1 / 32 / 128 |
| lr / max_len / steps | 1e-6 / 12288 / **19200** |
| pod | `mine-r926-…` GPUs **5,6** |
| launched | p4180 TRAIN pid**119797** + MERGE→n80 waiter |

Decision rule: Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
