# R1060 — cryptoDev SoftCtx MidRank Midβ Mega HiLR
Decision rule (pre-registered): Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36 (k=3, τ=0.03).

- base: `cryptoDev23/Affine-5Dku3dYp9j-hk8161@55b7ffe0`
- method: Offline-DPO Reason (teacher-side)
- knobs: β=0.1 α=128 r=32 lr=2e-6 @12288 epochs=4 max_steps=19200
- parent: p4189 teacher TP4→TP2 freed GPUs3,4; isolate Mega HiLR vs R1051 Mega MidLR
- pod: mine-r926 GPUs 3,4 → chall :8002
- p4209: first n80 died `EngineUnreachableError` mid-probe; merge kept; relaunch lean_chall p4209 → **CHALL_READY + n80 LIVE** sim pid133325
- **p4213 REFUTE v4** · m=**−0.015164** SE=0.004560 z=−3.325 n=80 bar≈0.00912 (~**−1.66×**) thought✓(252) B✓(0.377) k=3/τ=0.03 vs reign36 · teacher TP4 util0.80 GPUs0,1,5,6 · slot freed → R1051 n80 on same TKC
