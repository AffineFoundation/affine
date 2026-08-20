# R1091 — MidCtx LoRank Hiβ Hyper MidLR

**Status (p4223):** **TRAIN** on `mine-crown-1` GPUs 1,3 after R1066 REFUTE reap :8004.

| field | value |
|---|---|
| parent | R1066 MidCtx LoRank Hiβ Ultra HiLR REFUTE m=+0.004551 ~0.97× thought✓174 B✓0.403 k=3 → HyperExtra MidLR isolate; ≠ Ultra HiLR R1066 / ≠ ShortCtx LoRank Hiβ Hyper MidLR R1084 / ≠ MidCtx LoRank Midβ Ultra HiLR R1057 / ≠ SoftCtx LoRank Hiβ Ultra HiLR R967 / ≠ Online / ≠ GRPO |
| knobs | β=0.3 α=128 r=16 lr=1e-6 @8192 Soft Mid Mid Soft→ctx **max_steps=38400** |
| Decision rule | Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36 |
