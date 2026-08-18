# R766 — marsplan MidRank HiBeta MidCtx MegaSuperExtra ep4

**Axis:** Offline-DPO on `marsplan0624/affine-5gedzafcvg-queen`@`556d02a2`
β=0.3 · LoRA r=32/α=128 · lr=1e-6 · max_len=8192 · max_steps=19200 · epochs=4

**Why:** R754 HiRank HiBeta MidCtx Mega REFUTE m=−0.00862 ~−0.91× (thought✓238 B✓0.50).
MidRank sibling of the same HiBeta MidCtx Mega recipe; data from R723 MidRank HiBeta MidCtx.

**Decision rule:** Stage-5 iff fresh v4 n80 paired margin > max(2·SE, δ=0.002) AND median |z|≥80 AND B≥0.30 vs reign35.

**Pod:** zesty-comet-da GPUs 6,7 (after R754 chall reap). Leave R757 TRAIN on 4,5.
