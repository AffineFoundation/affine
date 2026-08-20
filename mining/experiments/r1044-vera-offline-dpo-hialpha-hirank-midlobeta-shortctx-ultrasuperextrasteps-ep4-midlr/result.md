# R1044 — ShortCtx HiRank MidLoβ Ultra MidLR (p4176)

## Claim
SoftCtx→ShortCtx isolate on HiRank+MidLoβ+Ultra+MidLR after R1029 SoftCtx Ultra MidLR REFUTE m=−0.001637 ~−0.29× vs reign36.

## Knobs
- base: `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`
- Offline-DPO Soft Mid Mid Soft→ShortCtx
- β=0.05 · α=128 · r=64 · lr=1e-6 · max_len=6144 · epochs=4 · max_steps=28800

## Decision rule
Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.

## Status
TRAIN armed p4176 on `mine-r938` GPUs 2,3 after exact-PID reap of stale R1029 chall :8002.
