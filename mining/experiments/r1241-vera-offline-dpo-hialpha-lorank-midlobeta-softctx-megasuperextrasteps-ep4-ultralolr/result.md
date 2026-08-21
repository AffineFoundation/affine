# R1241 — SoftCtx LoRank MidLoβ Mega UltraLoLR

| field | value |
|---|---|
| axis | `r1241-vera-offline-dpo-hialpha-lorank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr` |
| knobs | β=**0.05** α=128 r=**16** lr=**5e-7** @12288 Mega **max_steps=19200** |
| pod | r252 GPUs **6,7** |
| parent | R1207 SoftCtx HiRank MidLoβ Mega MidLR REFUTE m=+0.001811 ~0.44× (HiRank MidLoβ Mega LR exhausted R1020/R1040/R1207); R1202 Soft LoRank MidLoβ Mega MidLR ~−0.50× + R1229 HiLR TRAIN → UltraLoLR isolate |
| status | **TRAIN LIVE p4341** |
| decision | Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36 |
