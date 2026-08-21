# R1103 — MidCtx LoRank Hiβ Hyper HiLR

**Status (p4234):** **TRAIN** on `mine-crown-1` GPUs 1,3 after R1091 REFUTE reap :8004.

| field | value |
|---|---|
| parent | R1091 MidCtx LoRank Hiβ Hyper MidLR REFUTE m=+0.003552 ~0.46× thought✓165 B✓0.355 k=3 → HyperExtra HiLR isolate; ≠ MidLR R1091 / ≠ Ultra HiLR R1066 / ≠ ShortCtx LoRank Hiβ Hyper MidLR R1084 / ≠ MidCtx LoRank Midβ Ultra HiLR R1057 / ≠ SoftCtx LoRank Hiβ Ultra HiLR R967 / ≠ Online / ≠ GRPO |
| knobs | β=0.3 α=128 r=16 lr=2e-6 @8192 Soft Mid Mid Soft→ctx **max_steps=38400** |
| Decision rule | Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36 |
