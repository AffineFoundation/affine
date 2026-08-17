# R659 — ShortCtx MidRank HiBeta Mega ep3 × LoLR

**Axis:** r252 × Offline-DPO × HiAlpha × MidRank × HiBeta × ShortCtx × MegaExtra × epochs=3 × **LoLR (1e-6)**

**Why:** R637 Soft MidRank LoBeta ep3×LoLR SIGNAL ~1.45× rescued Soft LoBeta after ep2 fails. R622 Short MidRank HiBeta ep2 ~0.85× (best Short×HiBeta). R627 ep3@5e-6 REFUTE; R632 ep2×LoLR REFUTE. R659 is Short MidRank HiBeta ep3×LoLR — LoLR anti-overfit of R627 with R637's ep3×LoLR recipe (≠ R653 Short MidRank LoBeta / ≠ R658 Short MidRank MidBeta / ≠ R647 Short HiRank MidBeta).

**Knobs:** β=0.3 α=128 r=32 lr=**1e-6** @6144 max_steps=3600 epochs=**3**

**≠** R627 ep3@5e-6 / ≠ R632 ep2×LoLR / ≠ R622 ep2 / ≠ R653 Short MidRank LoBeta / ≠ R658 Short MidRank MidBeta / ≠ Soft MidRank HiBeta R651 / ≠ Online / ≠ GRPO R583

**Host:** mine-r226-marsplan-fullft-1 (brave) GPUs **2,3** (R653 on 0,1; R652 MERGE_DONE idle; R634 uplink live — train only, no dual-pipe)

**Decision rule:** Stage-5 iff fresh n80 paired margin > max(2·SE, 0.002) AND median |z|≥80 AND B pass≥0.30 vs **reign34** (v4 stamp k=3).
