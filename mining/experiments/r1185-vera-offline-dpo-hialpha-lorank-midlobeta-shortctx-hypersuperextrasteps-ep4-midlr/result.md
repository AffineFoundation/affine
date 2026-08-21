# R1185 — ShortCtx LoRank MidLoβ Hyper MidLR

**Status (p4301):** **TRAIN LIVE** on `mine-r339` GPUs **6,7** after R1166 REFUTE.

| field | value |
|---|---|
| parent | R1166 ShortCtx LoRank Midβ Hyper MidLR REFUTE m=+0.002389 SE=0.002645 ~0.45× thought✓181 B✓0.423 k=3 → MidLoβ MidLR isolate |
| knobs | β=**0.05** α=128 r=**16** lr=**1e-6** @6144 Soft Mid Mid Soft→ShortCtx **max_steps=38400** |
| decision | Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36 |

## Axes ≠
UltraLoLR R1146 / HiLR R1129 / Midβ MidLR R1166 / SoftCtx MidLoβ MidLR R1183 / HiRank MidLoβ MidLR R1167 / Online / GRPO

- **p4321:** REFUTE stamped; slot → R1206 Mega ShortCtx LoRank MidLoβ MidLR p4321
