# R1104 — MidCtx MidRank Loβ Hyper HiLR

**Status (p4234):** **TRAIN** on `mine-crown-1` GPUs 4,5 after R1093 REFUTE reap :8003.

| field | value |
|---|---|
| parent | R1093 MidCtx MidRank Loβ Hyper MidLR REFUTE m=+0.000646 ~0.10× thought✓179 B✓0.438 k=3 → HyperExtra HiLR isolate; ≠ MidLR R1093 / ≠ Ultra HiLR R1069 / ≠ MidCtx MidRank Loβ Ultra MidLR R1024 / ≠ SoftCtx MidRank Loβ Ultra HiLR R1043 / ≠ ShortCtx MidRank Loβ Ultra HiLR R1059 / ≠ Online / ≠ GRPO |
| knobs | β=0.02 α=128 r=32 lr=2e-6 @8192 Soft Mid Mid Soft→ctx **max_steps=38400** |
| Decision rule | Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36 |
