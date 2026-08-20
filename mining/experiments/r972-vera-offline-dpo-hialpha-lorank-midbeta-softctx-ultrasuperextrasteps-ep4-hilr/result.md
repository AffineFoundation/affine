# R972 — plan / result log

## Decision rule
Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.

## Axis
vera Offline-DPO HiAlpha **LoRank(r=16)** Midβ SoftCtx Ultra SuperExtra ep4 **HiLR(lr=2e-6)** @12288 max_steps=28800
Parent: R967 SoftCtx LoRank Hiβ Ultra HiLR REFUTE → Midβ isolate.

## p4094
- TRAIN on `mine-crown-1` GPUs **1,3**

## p4106
- MERGE idle→chall:8004 + v4 n80 LIVE

## p4107
- **REFUTE v4** m=**−0.011802** SE=0.006657 z=−1.773 n=80 bar≈0.01331 (~**−0.89×**) thought✓(198.5) B✓(0.50) k=3/τ=0.03 vs reign36
- exact-PID reap :8004 → **R983 SoftCtx LoRank Midβ Mega HiLR** TRAIN crown 1,3
