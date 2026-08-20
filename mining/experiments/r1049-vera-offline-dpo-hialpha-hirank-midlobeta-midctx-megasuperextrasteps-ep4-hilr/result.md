# R1049 — MidCtx HiRank MidLoβ Mega HiLR

## Claim
Mega HiLR isolate after R1038 MidCtx HiRank MidLoβ Mega UltraLoLR REFUTE.

## Knobs
- base: `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`
- Offline-DPO Soft Mid Mid Soft→MidCtx
- β=0.05 · α=128 · r=64 · lr=2e-6 · max_len=8192 · epochs=4 · max_steps=19200

## Decision rule
Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.

## Status
**REFUTE v4 (p4190)** on `mine-r338` vs reign36:
m=**−0.001495** SE=0.004770 z=−0.313 n=79 bar≈0.009540 (~**−0.16×**)
thought✓(214) B✓(0.557) k=3/τ=0.03 · chall reaped → **R1061** Midβ isolate.
