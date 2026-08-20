# R1054 — ShortCtx LoRank Hiβ Ultra MidLR

**Parent:** R1032 ShortCtx MidRank Hiβ Ultra MidLR CROWN_OK p4177
m=+0.005461 ~1.049× thought✓ B✓ k=3 vs reign36 (queued chal-00967).

**Axis:** vera Offline-DPO HiAlpha LoRank Hiβ ShortCtx Ultra MidLR
β=0.3 r=16 α=128 lr=1e-6 @6144 steps=28800 epochs=4

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.

**Pod:** mine-r924-vera-midctx-hibeta-1 GPUs 1,3 + MERGE→n80 waiter :8003 (p4183).
