# R1093 — MidCtx MidRank Loβ Hyper MidLR

**Status (p4223):** **TRAIN** on `mine-crown-1` GPUs 4,5 after R1069 REFUTE reap :8003.

| field | value |
|---|---|
| parent | R1069 MidCtx MidRank Loβ Ultra HiLR REFUTE m=-0.005219 ~-1.06× thought✓210.5 B✓0.538 k=3 → HyperExtra MidLR isolate; ≠ Ultra HiLR R1069 / ≠ MidCtx MidRank Loβ Ultra MidLR R1024 / ≠ SoftCtx MidRank Loβ Ultra HiLR R1043 / ≠ ShortCtx MidRank Loβ Ultra HiLR R1059 / ≠ Online / ≠ GRPO |
| knobs | β=0.02 α=128 r=32 lr=1e-6 @8192 Soft Mid Mid Soft→ctx **max_steps=38400** |
| Decision rule | Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36 |
