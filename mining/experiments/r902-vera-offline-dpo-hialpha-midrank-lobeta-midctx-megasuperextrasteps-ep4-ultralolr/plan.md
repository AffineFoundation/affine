# R902 — vera MidCtx Loβ Soft Mid Mid Soft UltraLoLR

**Claim:** MidCtx Loβ=0.02 MidRank on vera Soft Mid Mid Soft clears next crown vs reign36.
**Parent signal:** R885 SoftCtx Loβ REFUTE ~−0.05× + R886 SoftCtx MidLoβ ~0.27× → MidCtx Loβ transfer.
**Knobs:** β=0.02 α=128 r=32 lr=5e-7 max_len=8192 ep=4 max_steps=19200 Soft Mid Mid Soft.
**Decision rule:** Stage-5 iff fresh v4 n80 margin>max(2·SE,δ=0.002) AND thought≥80 AND B≥0.30 (k=3 τ=0.03).
**Pod:** mine-crown-1 / gentle-orbit-bd GPUs 4,5 (p4004 after R885 REFUTE).
