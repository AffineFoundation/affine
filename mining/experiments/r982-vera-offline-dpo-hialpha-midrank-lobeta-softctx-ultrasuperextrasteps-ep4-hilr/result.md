# R982 — plan / result log

## Decision rule
Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.

## Axis
vera Offline-DPO HiAlpha MidRank **Loβ** SoftCtx Ultra SuperExtra ep4 **HiLR(lr=2e-6)** @12288 max_steps=28800
Parent: R971 SoftCtx MidRank Midβ Ultra HiLR REFUTE m=+0.001843 ~0.21× → **Loβ isolate**.

## p4107
- TRAIN on `mine-crown-1` GPUs **4,5** after exact-PID reap R971 chall:8002
- TRAIN LIVE pid**137925** @2026-08-20T07:23:27Z

## p4120
- MERGE_DONE sat idle → chall+**v4 n80 LIVE** on crown (scripts `lean_chall_n80_crown_*_p4120.sh`)
- Ports: r981:8002 GPUs6,7 · r982:8003 GPUs4,5 · r983:8004 GPUs1,3
- Launched 2026-08-20T09:09:35Z
