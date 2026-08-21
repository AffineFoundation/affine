# R1186 — MidCtx HiRank MidLoβ Hyper MidLR

**Status (p4302):** **TRAIN LIVE** on `mine-r252` GPUs **6,7** after R1167 REFUTE.

| field | value |
|---|---|
| parent | R1167 ShortCtx HiRank MidLoβ Hyper MidLR REFUTE m=−0.000534 SE=0.001785 ~−0.15× thought✓171 B✓0.394 k=3 → MidCtx MidLR isolate |
| knobs | β=**0.05** α=128 r=**64** lr=**1e-6** @8192 Soft Mid Mid Soft→MidCtx **max_steps=38400** |
| decision | Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36 |

## Axes ≠
MidCtx UltraLoLR R1141 / SoftCtx MidLR R1090 / SoftCtx Hiβ UltraLoLR R1168 / ShortCtx MidLR R1167 / Online / GRPO
