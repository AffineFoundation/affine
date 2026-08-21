# R1168 — SoftCtx HiRank Hiβ Hyper UltraLoLR

## Decision rule (pre-registered)
Stage-5 iff fresh v4 n80 (k=3, τ=0.03) paired margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.

## Axis
- base: `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`
- Offline-DPO HiAlpha β=**0.3** r=64 α=128 lr=**5e-7** @12288 Soft Mid Mid Soft SoftCtx max_steps=38400 epochs=4
- Parent: **R1138 REFUTE** m=−0.004205 SE=0.002985 ~−0.70× thought✓166 B✓0.439 (SoftCtx HiRank MidLoβ UltraLoLR)
  MidLoβ SoftCtx LR family exhausted (MidLR R1090 / HiLR R1110 / UltraLoLR R1138) → **Hiβ UltraLoLR SoftCtx**
  (≠ MidLR re-run of R1090; SoftCtx Hiβ UltraLoLR never tried — Hiβ SoftCtx only HiLR R1114 / MidLR R1089)

## Status
- p4288: TRAIN LIVE on `mine-r252` GPUs 4,5 after exact-PID reap R1138 :8002
