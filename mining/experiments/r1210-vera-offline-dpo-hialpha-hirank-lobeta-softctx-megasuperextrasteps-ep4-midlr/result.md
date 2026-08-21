# R1210 — SoftCtx HiRank Loβ Mega MidLR (p4323)

| field | value |
|---|---|
| axis | `r1210-vera-offline-dpo-hialpha-hirank-lobeta-softctx-megasuperextrasteps-ep4-midlr` |
| knobs | β=**0.02** α=128 r=**64** lr=**1e-6** @12288 Mega **max_steps=19200** |
| pod | r337 GPUs **4,5** chall :8003 |
| parent | R1193 ShortCtx HiRank Loβ Hyper MidLR REFUTE m=-0.000439 SE=0.000986 ~-0.22x; ShortCtx HiRank Loβ LR exhausted (R1193 MidLR + R1197 UltraLoLR) → SoftCtx Mega MidLR isolate |
| status | **TRAIN LIVE p4323** |
| decision | Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36 |
