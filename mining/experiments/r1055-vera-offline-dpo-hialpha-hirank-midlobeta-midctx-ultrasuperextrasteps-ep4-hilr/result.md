# R1055 — MidCtx HiRank MidLoβ Ultra HiLR

**Parent:** R1033 MidCtx HiRank MidLoβ Ultra MidLR REFUTE p4177
m=+0.003590 ~0.83× thought✓ B✓ k=3 vs reign36.

**Axis:** vera Offline-DPO HiAlpha HiRank MidLoβ MidCtx Ultra HiLR
β=0.05 r=64 α=128 lr=2e-6 @8192 steps=28800 epochs=4

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.

**Pod:** mine-r252-vera-t4-nonking-grpo-1 GPUs 4,5 + MERGE→n80 waiter :8002 (p4185).
Stale R1030 chall :8002 reaped by exact PID before launch.

## Result p4204
**REFUTE v4** vs reign36 · m=**+0.000694** SE=0.002214 z=0.313 n=80 bar≈0.004428 (~**0.16×**)
thought✓(200.5) B✓(0.3875) k=3/τ=0.03 · → **R1076** SoftCtx isolate.
(Decision JSON note text was stale/wrong; chall_repo=`/tmp/r1055_merged`.)
