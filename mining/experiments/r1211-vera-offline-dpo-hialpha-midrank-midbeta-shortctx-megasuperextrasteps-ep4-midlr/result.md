# R1211 — ShortCtx MidRank Midβ Mega MidLR (p4323)

| field | value |
|---|---|
| axis | `r1211-vera-offline-dpo-hialpha-midrank-midbeta-shortctx-megasuperextrasteps-ep4-midlr` |
| knobs | β=**0.1** α=128 r=**32** lr=**1e-6** @6144 Mega **max_steps=19200** |
| pod | r337 GPUs **6,7** chall :8002 |
| parent | R1192 SoftCtx MidRank Midβ Hyper UltraLoLR REFUTE m=-0.001490 SE=0.001459 ~-0.51x; SoftCtx MidRank Midβ Mega MidLR/HiLR + Ultra MidLR exhausted (R1037/R977/R1023) → ShortCtx Mega MidLR isolate |
| status | **TRAIN LIVE p4323** |
| decision | Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36 |
