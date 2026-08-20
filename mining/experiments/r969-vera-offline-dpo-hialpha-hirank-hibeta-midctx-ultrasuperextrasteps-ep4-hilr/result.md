# R969 — plan / result log

## Decision rule
Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.

## Axis
vera Offline-DPO HiAlpha **HiRank(r=64)** Hiβ MidCtx Ultra SuperExtra ep4 **HiLR(lr=2e-6)** @8192 max_steps=28800
Parent: R953 UltraLoLR REFUTE m=+0.002543 ~0.47× → **HiLR isolate**.

## p4092
- TRAIN on `mine-r924` GPUs **4,5** pid**39526** after exact-PID reap R953 chall:8002

## p4108
- TRAIN_DONE → MERGE_DONE `/tmp/r969_merged` (16 shards) sat idle while R968 TRAIN on 6,7
- chall:8002 GPUs **4,5** UTIL=0.72 · Triton seed `chall_r953` n_so=26 · probe_ok
- **v4 n80 LIVE** pid**44647** → `/root/affine_data/r969_sim_result_reign36_wvk7.json`
- R968 TRAIN kept on 6,7; T:8000 K:8001 untouched
- Catalog: B300×8=0; sole 8×B200 BL `8f34559f` (skipped)
