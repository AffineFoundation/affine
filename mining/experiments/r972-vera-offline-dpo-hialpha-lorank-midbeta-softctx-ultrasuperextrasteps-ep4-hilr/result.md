# R972 — plan / result log

## Decision rule
Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.

## Axis
vera Offline-DPO HiAlpha **LoRank(r=16)** Midβ SoftCtx Ultra SuperExtra ep4 **HiLR(lr=2e-6)** @12288 max_steps=28800
Parent: R967 SoftCtx LoRank Hiβ Ultra HiLR REFUTE m=−0.002955 ~−0.52× → **Midβ isolate**.

## p4094
- TRAIN on `mine-crown-1` GPUs **1,3** after exact-PID reap R967 chall:8004
- TRAIN LIVE pid**124862** @2026-08-20T05:28:11Z

## p4106
- TRAIN_DONE @2026-08-20T06:59:33Z (1036 steps) → MERGE_DONE → `/tmp/r972_merged` @07:01:58Z `READY_FOR_N80`
- MERGE idle while R970/R971 n80 held :8002/:8003 → lean chall **:8004 GPUs 1,3** + **v4 n80 LIVE**
- Triton seed `chall_r971` n_so=26; probe_ok; sim pid**137062** → `/root/affine_data/r972_sim_result_reign36_wvk7.json`
- Decision rule unchanged (live crown rule, no headroom)
