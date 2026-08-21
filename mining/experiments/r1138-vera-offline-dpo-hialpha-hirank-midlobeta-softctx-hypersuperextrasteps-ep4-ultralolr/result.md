# R1138 — SoftCtx HiRank MidLoβ Hyper UltraLoLR

## Decision rule (pre-registered)
Stage-5 iff fresh v4 n80 (k=3, τ=0.03) paired margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.

## Axis
- base: `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`
- Offline-DPO HiAlpha β=0.05 r=64 α=128 lr=**5e-7** @12288 Soft Mid Mid Soft SoftCtx max_steps=38400 epochs=4
- Parent: **R1110 REFUTE** m=−0.006349 SE=0.005119 ~−0.62× thought✓188 B✓0.464 (Hyper HiLR) → UltraLoLR isolate
  (MidLR R1090 and HiLR R1110 both failed on SoftCtx HiRank MidLoβ Hyper)

## Status
- p4262: TRAIN LIVE on `mine-r252` GPUs 4,5 after exact-PID reap R1110 :8002
