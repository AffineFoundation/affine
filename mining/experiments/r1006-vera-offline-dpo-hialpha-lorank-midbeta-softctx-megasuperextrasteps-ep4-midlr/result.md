# R1006 — plan / result log

## Decision rule
Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.

## Axis
vera Offline-DPO HiAlpha **LoRank(r=16)** Midβ SoftCtx **Mega** SuperExtra ep4 **MidLR(lr=1e-6)** @12288 max_steps=**19200**
Parent: R993 SoftCtx LoRank Midβ Mega UltraLoLR REFUTE m=+0.002678 ~0.58× → **MidLR isolate** (between R983 HiLR and R993 UltraLoLR).

## p4139
- exact-PID reap R993 chall:8004
- TRAIN on `mine-crown-1` GPUs **1,3**
- wait→merge + MERGE→n80 waiter armed
