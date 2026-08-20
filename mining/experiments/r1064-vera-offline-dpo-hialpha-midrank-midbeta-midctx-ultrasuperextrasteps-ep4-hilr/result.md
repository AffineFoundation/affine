# R1064 — MidCtx MidRank Midβ Ultra HiLR

**Status:** TRAIN p4194 on `mine-r337` GPUs **4,5** pid**121458** + MERGE→n80 waiter (:8003).

**Parent:** R1047 MidCtx MidRank MidLoβ Ultra HiLR REFUTE m=+0.002013 ~0.19× → Midβ isolate.

**Knobs:** base `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`, β=0.1, α=128, r=32, lr=2e-6, max_len=8192, epochs=4, max_steps=28800.

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
