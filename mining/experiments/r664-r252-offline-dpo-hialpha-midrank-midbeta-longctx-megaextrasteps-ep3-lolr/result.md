# R664 — in flight

## Status (p3681)
- **TRAIN** crown GPUs **2,3** pid**13901** · Long MidRank MidBeta Mega ep3×LoLR · kept=604
- wait→merge armed (`wait_r664_train_then_merge_p3681.sh`)
- Knobs: β=0.1 · α=128 · r=32 · lr=1e-6 · @16384 · max_steps=3600 · ep=3
- Parent: R641 ~0.31× ep2; sibling R637 Soft MidRank LoBeta ep3 SIGNAL ~1.45×

## Decision rule (pre-registered)
Stage-5 iff fresh n80 paired margin > max(2·SE, 0.002) AND median |z|≥80 AND B pass≥0.30 vs reign34 (v4 k=3).
