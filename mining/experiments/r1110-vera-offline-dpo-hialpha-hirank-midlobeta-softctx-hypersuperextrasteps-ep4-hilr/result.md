# R1110 — SoftCtx HiRank MidLoβ Hyper HiLR

## Decision rule (pre-registered)
Stage-5 iff fresh v4 n80 (k=3, τ=0.03) paired margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.

## Axis
- base: `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`
- Offline-DPO HiAlpha β=0.05 r=64 α=128 lr=2e-6 @12288 Soft Mid Mid Soft SoftCtx max_steps=38400 epochs=4
- Parent: **R1090 REFUTE** m=−0.001520 SE=0.001362 ~−0.56× thought✓187 B✓0.466 (Hyper MidLR) → Hyper HiLR isolate

## Status
- p4239: TRAIN LIVE on `mine-r252` GPUs 4,5 after exact-PID reap R1090 :8002
