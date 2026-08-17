# R654 — MidCtx MidRank LoBeta Mega ep3 × LoLR

**Axis:** r252 × Offline-DPO × HiAlpha × MidRank × LoBeta × MidCtx × MegaExtra × epochs=3 × **LoLR (1e-6)**

**Why:** R637 Soft MidRank LoBeta ep3×LoLR SIGNAL ~1.45× rescued Soft LoBeta after ep2 fails. R639 MidCtx MidRank LoBeta ep2×LoLR is MERGE_DONE (pending SCP). R586 MidCtx MidRank LoBeta ep1 ~0.27×. R654 is the MidCtx MidRank LoBeta ep3×LoLR sibling of R637 (≠ R639 ep2 / ≠ Soft MidRank LoBeta R637 / ≠ Short MidRank LoBeta R653 / ≠ MidCtx MidRank MidBeta R642).

**Knobs:** β=0.02 α=128 r=32 lr=**1e-6** @8192 max_steps=3600 epochs=**3**

**≠** R639 ep2×LoLR / ≠ R615 ep2 @5e-6 / ≠ R586 ep1 / ≠ Soft MidRank LoBeta R637 / ≠ Short MidRank LoBeta R653 / ≠ MidCtx MidRank MidBeta R642 / ≠ Online / ≠ GRPO R583

**Host:** mine-crown-1 (gentle-orbit-bd) GPUs **0,1** (all GPUs idle; B300 stock empty; R634 uplink on brave)

**Decision rule:** Stage-5 iff fresh n80 paired margin > max(2·SE, 0.002) AND median |z|≥80 AND B pass≥0.30 vs **reign34** (v4 stamp k=3).
