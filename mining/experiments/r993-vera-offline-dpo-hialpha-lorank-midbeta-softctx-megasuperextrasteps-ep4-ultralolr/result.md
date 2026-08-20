# R993 — plan / result log

## Decision rule
Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.

## Axis
vera Offline-DPO HiAlpha **LoRank(r=16)** Midβ SoftCtx **Mega** SuperExtra ep4 **UltraLoLR(lr=5e-7)** @12288 max_steps=**19200**
Parent: R983 SoftCtx LoRank Midβ Mega HiLR REFUTE m=−0.008220 ~−0.87× → **UltraLoLR isolate** (mirrors R959 Mega UltraLoLR CROWN on LoRank).

## p4123
- exact-PID reap R983 chall:8004 pid**143006**
- TRAIN on `mine-crown-1` GPUs **1,3** pid**151338** @2026-08-20T09:28:01Z
- wait→merge armed pid**151343**

## p4137
- TRAIN_DONE @2026-08-20T10:58:49Z steps=**1036** adapter `/root/r993/train/adapter`
- MERGE live on GPUs **1,3** (merge_lora pid**154773**) → `/tmp/r993_merged`
- **MERGE→n80** waiter armed pid**155035** → lean chall **:8004** GPUs 1,3 vs reign36 wvk7
- scripts: `wait_r993_merge_then_n80_p4137.sh` + `lean_chall_n80_crown_r993_gpus13_p4137.sh`
