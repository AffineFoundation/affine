# R1044 — ShortCtx HiRank MidLoβ Ultra MidLR (p4176)

## Claim
SoftCtx→ShortCtx isolate on HiRank+MidLoβ+Ultra+MidLR after R1029 SoftCtx Ultra MidLR REFUTE m=−0.001637 ~−0.29× vs reign36.

## Knobs
- base: `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`
- Offline-DPO Soft Mid Mid Soft→ShortCtx
- β=0.05 · α=128 · r=64 · lr=1e-6 · max_len=6144 · epochs=4 · max_steps=28800

## Decision rule
Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.

## Result (p4191)
**REFUTE v4** vs reign36 · m=**+0.000182** SE=0.001229 z=0.149 n=79 bar≈0.002457 (~**0.07×**)
thought✓(167) B✓(0.416) k=3/τ=0.03 · chall reaped → **R1062** Ultra HiLR.
