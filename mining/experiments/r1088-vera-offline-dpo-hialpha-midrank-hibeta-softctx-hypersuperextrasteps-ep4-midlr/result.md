# R1088 — SoftCtx isolate after R1077 ShortCtx Hyper MidLR REFUTE

## Decision rule (pre-registered)
Stage-5 submit iff fresh **v4** n80 (k=3, τ=0.03) vs live reign36 clears
`margin > max(2·SE, δ=0.002)` **and** median stripped `|z| ≥ 80` **and** B pass ≥ 0.30.

## Parent signal (p4221)
**R1077 REFUTE v4** vs reign36: m=+0.001670 SE=0.002622 z=0.637 n=80
bar≈0.005244 (~**0.32×**) thought✓(168) B✓(0.361) k=3/τ=0.03.
Axis was ShortCtx MidRank Hiβ Hyper MidLR @6144.

## Axis
`vera×Offline-DPO×HiAlpha×MidRank×HiBeta×SoftCtx×HyperSuperExtraSteps×epochs=4×MidLR`
- base `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`
- β=0.3 α=128 r=32 lr=1e-6 max_len=12288 max_steps=38400
- ≠ ShortCtx R1077 / ≠ SoftCtx Midβ Hyper MidLR R1083 / ≠ MidCtx Hiβ Hyper MidLR R1085

## Ops
- pod `mine-r338-marsplan-online-dpo-bigg-hilr-1` SSH `95.133.253.90:40099`
- GPUs **6,7** after exact-PID reap of R1077 chall :8002
- keep R1087 TRAIN on GPUs 4,5
