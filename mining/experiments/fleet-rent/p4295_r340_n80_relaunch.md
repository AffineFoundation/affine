# R1142 / R1144 / R1156 — p4295 r340 n80 relaunch

**When:** 2026-08-21T10:13:53Z
**Pod:** `mine-r340-marsplan-online-dpo-hirank-bigg-1` (`gentle-orbit-4a`) SSH `18.118.83.97:40127`

## What was wrong
- **R1142** (GPUs1,2 TP2 :8002) and **R1144** (GPUs6,7 TP2 :8004): vLLM workers stalled after NCCL init (~1 GiB VRAM, API dead, never `CHALL_READY`) since ~08:17 / ~08:39.
- **R1156** (GPU3 TP1 util0.90 :8003): reached CHALL_READY, then CUDA OOM on prompt logprobs → `FATAL missing sim result` @09:12.

## Fix (p4295)
- Kill orphan trees **by PID** (never `pkill -f`).
- Patch lean → **TP1 util=0.85**, FORCE Triton wipe+seed from `king`, `/v1/completions` probe before n80.
- Relaunch: R1142 GPU**1** :8002 · R1144 GPU**6** :8004 · R1156 GPU**3** :8003.
- Scripts: `fleet-rent/patch_lean_r340_p4295.py`, `fleet-rent/p4295_relaunch_stuck_n80_r340_r1142_r1144_r1156.sh`.

## Status at arm
Challengers loading (~68 GiB on GPUs 1/3/6 @10:14Z). Poll `/root/logs/p4295_*_chall_n80_relaunch.nohup` + sim JSON under `/root/affine_data/r11{42,44,56}_*_wvk7.json`.
