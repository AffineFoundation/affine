# R938 — vera SoftCtx MidRank Hiβ

**Axis:** Offline-DPO Soft Mid Mid Soft · β=0.3 · r=32 · α=128 · lr=5e-7 · @12288 · epochs=4 · max_steps=19200
**Base:** `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`
**Pod:** `mine-r938-vera-softctx-hibeta-1` noble-wolf-22 8×H200 $15.96/h · GPUs 0,1 · TTL 24h
**Why:** MidCtx Hiβ (R924) + ShortCtx Hiβ REFUTE ~0.56× (R923) → SoftCtx MidRank Hiβ isolate
**Decision:** Stage-5 iff fresh v4 n80 margin>max(2·SE,0.002) AND thought≥80 AND B≥0.30 vs reign36
