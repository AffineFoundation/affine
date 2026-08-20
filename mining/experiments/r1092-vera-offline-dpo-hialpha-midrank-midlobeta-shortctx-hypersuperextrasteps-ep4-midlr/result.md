# R1092 — ShortCtx MidRank MidLoβ Hyper MidLR

**Status (p4223):** **TRAIN** on `mine-crown-1` GPUs 6,7 after R1067 REFUTE reap :8002.

| field | value |
|---|---|
| parent | R1067 ShortCtx MidRank MidLoβ Ultra HiLR REFUTE m=-0.003970 ~-0.74× thought✓199 B✓0.492 k=3 → HyperExtra MidLR isolate; ≠ Ultra HiLR R1067 / ≠ ShortCtx MidRank MidLoβ Mega MidLR R1042 / ≠ ShortCtx HiRank MidLoβ Hyper MidLR R1079 / ≠ ShortCtx MidRank MidLoβ Mega HiLR R1056 / ≠ Online / ≠ GRPO |
| knobs | β=0.05 α=128 r=32 lr=1e-6 @6144 Soft Mid Mid Soft→ctx **max_steps=38400** |
| Decision rule | Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36 |
