# R1244 — SoftCtx LoRank Loβ Mega HiLR (single GPU)

| field | value |
|---|---|
| axis | `r1244-vera-offline-dpo-hialpha-lorank-lobeta-softctx-megasuperextrasteps-ep4-hilr` |
| knobs | β=**0.02** α=128 r=**16** lr=**2e-6** @12288 Mega **max_steps=19200** |
| pod | r1158 GPU **1** (single) |
| parent | R1236 SoftCtx LoRank Loβ Mega MidLR TRAIN → Mega HiLR isolate; cell was missing |
| status | **TRAIN LIVE p4342** |
| decision | Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36 |
