# R1242 — ShortCtx LoRank Midβ Mega HiLR

| field | value |
|---|---|
| axis | `r1242-vera-offline-dpo-hialpha-lorank-midbeta-shortctx-megasuperextrasteps-ep4-hilr` |
| knobs | β=**0.1** α=128 r=**16** lr=**2e-6** @6144 Mega **max_steps=19200** |
| pod | r1158 GPUs **4,5** |
| parent | R1235 ShortCtx LoRank Midβ Mega MidLR TRAIN (R1166 Hyper MidLR REFUTE ~0.45×) → Mega HiLR isolate; cell was missing |
| status | **TRAIN LIVE p4342** |
| decision | Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36 |
