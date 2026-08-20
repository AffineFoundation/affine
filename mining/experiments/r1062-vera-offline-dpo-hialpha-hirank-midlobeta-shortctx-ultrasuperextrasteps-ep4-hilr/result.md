# R1062 — ShortCtx HiRank MidLoβ Ultra HiLR (p4191)

## Claim
R1044 ShortCtx HiRank MidLoβ Ultra MidLR REFUTE m=+0.000182 ~0.07×
(thought✓167 B✓0.416 k=3) → MidLR→HiLR isolate on same ShortCtx Ultra lane.

## Knobs
- base: `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`
- Offline-DPO Soft Mid Mid Soft→ShortCtx
- β=0.05 · α=128 · r=64 · lr=**2e-6** · max_len=6144 · epochs=4 · max_steps=28800

## ≠
MidLR R1044 / ShortCtx Mega HiLR R1058 / SoftCtx Ultra MidLR R1029 /
MidCtx Ultra HiLR R1055 / SoftCtx Mega HiLR R1040 / Online / GRPO

## Decision rule
Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.

## Status
TRAIN armed p4191 on `mine-r938` GPUs 2,3 after exact-PID reap of R1044 chall :8002.
