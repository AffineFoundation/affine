# R1071 — ShortCtx HiRank Hiβ Ultra HiLR

**Status (p4201):** **TRAIN** on `mine-r339` GPUs 6,7 after R1052 REFUTE reap :8003.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.3** / 128 / **64** / **2e-6** |
| max_len / steps | **6144** / **28800** |
| parent | R1052 ShortCtx HiRank Hiβ Ultra MidLR REFUTE m=−0.001866 SE=0.002315 z=−0.806 n=78 bar≈0.004630 (~**−0.40×**) thought✓(152) B✓(0.397) k=3/τ=0.03 → HiLR isolate |

≠ MidLR R1052 / ≠ MidRank ShortCtx Hiβ Ultra HiLR R1065 / ≠ SoftCtx HiRank Hiβ Ultra MidLR R1045 / ≠ R1032 CROWN MidRank / ≠ Online / ≠ GRPO

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
