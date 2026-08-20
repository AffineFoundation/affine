# R1020 — SoftCtx HiRank MidLoβ Mega MidLR

**Parent:** R998 SoftCtx HiRank MidLoβ Mega UltraLoLR REFUTE p4154
m=+0.001298 SE=0.001968 bar≈0.003937 (~0.33×) thought✓171.5 B✓0.4125 k=3 vs reign36.

**Axis:** vera Offline-DPO HiAlpha HiRank MidLoβ SoftCtx Mega MidLR
β=0.05 r=64 α=128 lr=1e-6 @12288 steps=19200 epochs=4

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.

**Pod:** mine-r252-vera-t4-nonking-grpo-1 GPUs 6,7 + MERGE→n80 waiter (p4154).

## Result p4170 — REFUTE v4
- margin=+0.001734 SE=0.002489 z=0.696 n=80 bar≈0.004978 (~0.35×)
- thought_median=169.5 ✓ · B pass=0.2875 ✗ (causality_fail) · k=3 τ=0.03
- Follow-on: **R1040** SoftCtx HiRank MidLoβ Mega HiLR (lr=2e-6) on same GPUs 6,7
