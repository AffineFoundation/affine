# R1119 — ShortCtx MidRank Midβ Hyper HiLR

**Status (p4250):** **TRAIN** after R1107 REFUTE reap on `mine-r337` GPUs 4,5.

| field | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| method | Offline-DPO HiAlpha MidRank Midβ ShortCtx HyperSuperExtra ep4 HiLR |
| knobs | β=0.1 α=128 r=32 lr=2e-6 @6144 max_steps=38400 |
| parent | R1107 MidCtx MidRank Midβ Hyper HiLR REFUTE m=+0.004879 ~0.66× thought✓181 B✓0.456 |
| decision | Stage-5 iff v4 n80 margin>max(2·SE,δ=0.002) AND thought≥80 AND B≥0.30 vs reign36 |
