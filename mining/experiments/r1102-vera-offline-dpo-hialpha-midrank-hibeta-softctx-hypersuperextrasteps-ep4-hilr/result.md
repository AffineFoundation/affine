# R1102 — SoftCtx MidRank Hiβ HyperExtra HiLR

**Status (p4252):** **REFUTE v4** vs reign36 → **R1124** ShortCtx MidRank Hiβ Hyper HiLR.

## Decision rule (pre-registered)
Stage-5 submit iff fresh **v4** n80 (k=3, τ=0.03) vs live reign36 clears
`margin > max(2·SE, δ=0.002)` **and** median stripped `|z| ≥ 80` **and** B pass ≥ 0.30.

## Parent signal (p4233)
**R1088 REFUTE v4** vs reign36: m=+0.001775 SE=0.001280 z=1.386 n=80
bar≈0.002560 (~**0.69×**) thought✓(168) B✓(0.45) k=3/τ=0.03.
Axis was SoftCtx MidRank Hiβ Hyper MidLR @12288.

## n80 (p4252 scrape; sim DONE 2026-08-21T02:58:01Z)
| field | value |
|---|---|
| n80 | m=**−0.004769** SE=0.004243 z=−1.124 n=80 bar≈0.008485 (~**−0.56×**) thought✓(189) B✓(0.4875) k=3/τ=0.03 |

## Axis
`vera×Offline-DPO×HiAlpha×MidRank×HiBeta×SoftCtx×HyperSuperExtraSteps×epochs=4×HiLR`
- base `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`
- β=0.3 α=128 r=32 lr=**2e-6** max_len=12288 max_steps=38400

## Ops
- pod `mine-r338` GPUs **6,7** :8002 reaped p4252 → **R1124** TRAIN
