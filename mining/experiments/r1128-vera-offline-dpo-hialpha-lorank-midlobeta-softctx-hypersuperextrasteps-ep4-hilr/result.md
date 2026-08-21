# R1128 — result

**Axis:** vera SoftCtx LoRank MidLoβ HyperSuperExtra ep4 HiLR (β=0.05 r=16 lr=2e-6 @12288 steps=38400)
**Parent:** R1048-family SoftCtx LoRank isolate
**Pod:** mine-r340 (gentle-orbit-4a) GPUs 6,7 → TP1 GPU6

## Status (p4269)
- TRAIN DONE → MERGE DONE (`/tmp/r1128_merged` 16 shards)
- p4268: TP1 util**0.90** CHALL_READY + n80 pid65193 → **OOM** on prompt logprobs (~+7.58GiB) → EngineDead → n80 ConnectError
- p4269: re-arm TP1 util**0.85** :8004 pid**65916** CHALL_READY ~85s + n80 pid**67441** (alive past probe @06:21Z)
- Decision: crown if paired margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 (wvk=7)

## Check
- `tail -f /root/logs/p4269_r1128_chall_n80_wvk7.log`
- `cat /root/affine_data/r1128_sim_progress_reign36_wvk7.json`
- result: `/root/affine_data/r1128_sim_result_reign36_wvk7.json`
