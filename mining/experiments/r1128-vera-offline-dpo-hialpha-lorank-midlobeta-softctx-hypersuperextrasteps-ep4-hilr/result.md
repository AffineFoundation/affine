# R1128 — SoftCtx LoRank MidLoβ Hyper HiLR (fill idle r340 6,7)
| field | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| knobs | β=0.05 α=128 r=**16** lr=2e-6 @12288 Soft Mid Mid Soft **max_steps=38400** |
| parent | R340 Online-DPO aborted → idle GPUs6,7; LoRank isolate vs MidRank R1120 / HiRank R1110 |
| decision | Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36 |
| train | DONE 2026-08-21T06:04Z · merge DONE 06:06Z shards=16 `/tmp/r1128_merged` |
| p4268 | lean polled wrong GPUs1,2 → stuck; TP2 stall VRAM≈1GiB → **TP1** GPU6 :8004 CHALL_READY + n80 pid**65193** |
