# R1166 — ShortCtx LoRank Midβ Hyper MidLR

**Status (p4286):** **TRAIN LIVE** on `mine-r339` GPUs 6,7 after R1145 REFUTE reap.

| field | value |
|---|---|
| parent | R1145 ShortCtx LoRank Midβ Hyper UltraLoLR REFUTE m=−0.001512 SE=0.002307 ~−0.33× thought✓166 B✓0.428 k=3 → MidLR isolate |
| knobs | β=0.1 α=128 r=**16** lr=**1e-6** @6144 Soft Mid Mid Soft→ShortCtx **max_steps=38400** |
| train pid | **75354** · SSH `23.153.44.20:40299` |
| decision | Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36 |

## Axes ≠
UltraLoLR R1145 / HiLR R1127 / MidCtx LoRank Midβ MidLR R1163 / SoftCtx LoRank Midβ UltraLoLR R1149 / Online / GRPO
