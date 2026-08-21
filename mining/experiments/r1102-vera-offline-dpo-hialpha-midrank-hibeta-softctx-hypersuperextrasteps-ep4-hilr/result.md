# R1102 — Hyper HiLR isolate after R1088 SoftCtx MidRank Hiβ Hyper MidLR near-miss

## Decision rule (pre-registered)
Stage-5 submit iff fresh **v4** n80 (k=3, τ=0.03) vs live reign36 clears
`margin > max(2·SE, δ=0.002)` **and** median stripped `|z| ≥ 80` **and** B pass ≥ 0.30.

## Parent signal (p4233)
**R1088 REFUTE v4** vs reign36: m=+0.001775 SE=0.001280 z=1.386 n=80
bar≈0.002560 (~**0.69×**) thought✓(168) B✓(0.45) k=3/τ=0.03.
Axis was SoftCtx MidRank Hiβ Hyper MidLR @12288.

## Axis
`vera×Offline-DPO×HiAlpha×MidRank×HiBeta×SoftCtx×HyperSuperExtraSteps×epochs=4×HiLR`
- base `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`
- β=0.3 α=128 r=32 lr=**2e-6** max_len=12288 max_steps=38400
- ≠ MidLR R1088 / ≠ SoftCtx MidRank Hiβ Ultra HiLR R1050 / ≠ SoftCtx MidRank Hiβ Ultra MidLR R1039 / ≠ SoftCtx MidRank Midβ Hyper MidLR R1083 / ≠ MidCtx MidRank Hiβ Hyper MidLR R1085 / ≠ Online / ≠ GRPO

## Ops
- pod `mine-r338-marsplan-online-dpo-bigg-hilr-1` SSH `95.133.253.90:40099`
- GPUs **6,7** after exact-PID reap of R1088 chall :8002
- keep R1100 TRAIN on GPUs 4,5
