# R1079 — ShortCtx HiRank MidLoβ HyperExtra MidLR

**Status:** **REFUTE v4** p4216 on `mine-r938` GPUs **2,3** → cascaded **R1082**.

**Parent:** R1062 ShortCtx HiRank MidLoβ Ultra HiLR REFUTE m=+0.002051 ~0.18× → MidLR restore + HyperExtra 38400.

**Knobs:** base `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`, β=0.05, α=128, r=64, lr=1e-6, max_len=6144, epochs=4, max_steps=38400.

**n80 vs reign36 (wvk=7, k=3, τ=0.03):** m=**−0.001769** SE=0.001183 z=−1.495 n=78 bar≈0.002366 (~**−0.75×**) thought✓(158.5) B✓(0.410).

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36 — **failed**.
