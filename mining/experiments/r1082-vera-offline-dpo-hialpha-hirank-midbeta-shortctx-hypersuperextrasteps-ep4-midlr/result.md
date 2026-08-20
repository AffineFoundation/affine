# R1082 — ShortCtx HiRank Midβ HyperExtra MidLR

**Status:** TRAIN p4216 on `mine-r938` GPUs **2,3** after R1079 REFUTE.

**Parent:** R1079 ShortCtx HiRank MidLoβ HyperExtra MidLR REFUTE
m=−0.001769 SE=0.001183 z=−1.495 n=78 bar≈0.002366 (~−0.75×)
thought✓158.5 B✓0.410 k=3/τ=0.03 vs reign36 → Midβ isolate (β 0.05→0.1).

**Knobs:** base `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`, β=**0.1**, α=128, r=64, lr=1e-6, max_len=6144, epochs=4, max_steps=38400.

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
