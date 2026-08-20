# R970 — plan / result log

## Decision rule
Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.

## Axis
vera Offline-DPO HiAlpha MidRank MidLoβ SoftCtx Ultra SuperExtra ep4 **HiLR(lr=2e-6)** @12288 max_steps=28800
Parent: R966 MidCtx MidRank MidLoβ Ultra HiLR REFUTE → SoftCtx isolate.

## p4093
- TRAIN on `mine-crown-1` GPUs **6,7**

## p4105
- MERGE idle→chall:8003 + v4 n80 LIVE

## p4107
- **REFUTE v4** m=**−0.002056** SE=0.004982 z=−0.413 n=79 bar≈0.00996 (~**−0.21×**) thought✓(195) B✓(0.488) k=3/τ=0.03 vs reign36
- exact-PID reap :8003 → **R981 SoftCtx MidRank MidLoβ Mega HiLR** TRAIN crown 6,7
