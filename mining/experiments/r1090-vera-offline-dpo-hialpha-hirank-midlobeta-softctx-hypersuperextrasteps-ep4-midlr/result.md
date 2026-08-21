# R1090 — SoftCtx HiRank MidLoβ Hyper MidLR

## Decision rule (pre-registered)
Stage-5 iff fresh v4 n80 (k=3, τ=0.03) paired margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.

## Axis
- base: `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`
- Offline-DPO HiAlpha β=0.05 r=64 α=128 lr=1e-6 @12288 Soft Mid Mid Soft SoftCtx max_steps=38400 epochs=4
- Parent: **R1076 REFUTE** m=−0.001981 SE=0.005687 ~−0.17× thought✓195 B✓0.459 (Ultra HiLR) → HyperExtra MidLR isolate

## Status
- p4222: TRAIN LIVE on `mine-r252` GPUs 4,5 after reap R1076 :8002
- **p4239 REFUTE v4** m=**−0.001520** SE=0.001362 z=−1.117 n=80 bar≈0.002723 (~**−0.56×**) thought✓(187) B✓(0.466) k=3/τ=0.03 vs **reign36** · → **R1110** Hyper HiLR
