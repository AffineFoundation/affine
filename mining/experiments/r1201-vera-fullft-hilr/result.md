# R1201 — result notes

## Status
- **p4320:** p4319 n80 pid42003 died — teacher OOM on prompt_logprobs at util**0.90**. Kill-by-PID T/K/chall → `serve_three` TP1 **GPUUTIL=0.85** **CHALL_GPUUTIL=0.80** → `/v1/models` + completions+logprobs probe OK → n80 relaunch pid**48168** → `/root/affine_data/r1201_sim_result_reign36_wvk7.json`. SSH `69.63.236.163:40299`.
- **p4317:** After **R1191 REFUTE ~0.02×**, freed T/K/chall by PID on `mine-r1191` H200; launched **FullFT HiLR** lr=**2e-6** ep=1 @8192 on `winner_za_high_l2` (406). TRAIN done; merge local (`SKIP_HF_PUSH=1`).

## Axis
vera×FullFT×HiLR isolate after MidLR REFUTE. ≠ R1191 MidLR · ≠ Offline-DPO · ≠ Online-DPO · ≠ R1158 GRPO.

## Decision
Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.
