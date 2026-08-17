# R665 — in flight

## Status (p3682)
- **TRAIN** crown GPUs **4,5** · Long HiRank MidBeta Mega ep3×LoLR · kept=604
- wait→merge armed
- Knobs: β=0.1 · α=128 · r=64 · lr=1e-6 · @16384 · max_steps=3600 · ep=3
- Parent: R646 ~−0.49× ep2; sibling R637 Soft MidRank LoBeta ep3 SIGNAL ~1.45×

## Decision rule (pre-registered)
Stage-5 iff fresh n80 paired margin > max(2·SE, 0.002) AND median |z|≥80 AND B pass≥0.30 vs reign34 (v4 k=3).
