# R971 — plan / result log

## Decision rule
Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.

## Axis
vera Offline-DPO HiAlpha MidRank Midβ SoftCtx Ultra SuperExtra ep4 **HiLR(lr=2e-6)** @12288 max_steps=28800
Parent: R965 ShortCtx MidRank Midβ Ultra HiLR REFUTE → SoftCtx isolate.

## p4093
- TRAIN on `mine-crown-1` GPUs **4,5**

## p4105
- MERGE idle→chall:8002 + v4 n80 LIVE

## p4107
- **REFUTE v4** m=**+0.001843** SE=0.004409 z=0.418 n=79 bar≈0.00882 (~**0.21×**) thought✓(183) B✓(0.468) k=3/τ=0.03 vs reign36
- exact-PID reap :8002 → **R982 SoftCtx MidRank Loβ Ultra HiLR** TRAIN crown 4,5
