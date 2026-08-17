# R656 — ShortCtx HiRank LoBeta Mega ep3 × LoLR

**Axis:** r252 × Offline-DPO × HiAlpha × HiRank × LoBeta × ShortCtx × MegaExtra × epochs=3 × **LoLR (1e-6)**

**Why:** R637 Soft MidRank LoBeta ep3×LoLR SIGNAL ~1.45× rescued Soft LoBeta after ep2 fails. R617 Short HiRank LoBeta ep2 @5e-6 ~0.64×; R628 same cell ep3 @5e-6 REFUTE. R634 Short HiRank LoBeta ep2×LoLR is SCP→n80. R656 is the Short HiRank LoBeta ep3×LoLR sibling of R637 (LoLR may avoid R628's 5e-6 collapse).

**Knobs:** β=0.02 α=128 r=64 lr=**1e-6** @6144 max_steps=3600 epochs=**3**

**≠** R634 ep2×LoLR / ≠ R628 ep3 @5e-6 / ≠ R617 ep2 @5e-6 / ≠ Short MidRank LoBeta R653 / ≠ MidCtx HiRank LoBeta R655 / ≠ Soft HiRank LoBeta R652 / ≠ Online / ≠ GRPO R583

**Host:** mine-crown-1 (gentle-orbit-bd) GPUs **2,3** (0,1=R654; B300 stock empty; R634 uplink on brave→zesty)

**Decision rule:** Stage-5 iff fresh n80 paired margin > max(2·SE, 0.002) AND median |z|≥80 AND B pass≥0.30 vs **reign34** (v4 stamp k=3).
