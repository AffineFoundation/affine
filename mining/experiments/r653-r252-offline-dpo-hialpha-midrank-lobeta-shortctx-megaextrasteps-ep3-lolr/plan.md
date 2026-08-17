# R653 — ShortCtx MidRank LoBeta Mega ep3 × LoLR

**Axis:** r252 × Offline-DPO × HiAlpha × MidRank × LoBeta × ShortCtx × MegaExtra × epochs=3 × **LoLR (1e-6)**

**Why:** R637 Soft MidRank LoBeta ep3×LoLR SIGNAL ~1.45× rescued Soft LoBeta after ep2 fails. R638 Short MidRank LoBeta ep2×LoLR is ARMED (pending SCP). R590 Short MidRank LoBeta ep1 ~0.50×. R653 is the Short MidRank LoBeta ep3×LoLR sibling of R637 (≠ R638 ep2 / ≠ Soft MidRank LoBeta R637 / ≠ Short HiRank LoBeta R634).

**Knobs:** β=0.02 α=128 r=32 lr=**1e-6** @6144 max_steps=3600 epochs=**3**

**≠** R638 ep2×LoLR / ≠ R614 ep2 @5e-6 / ≠ R590 ep1 / ≠ Soft MidRank LoBeta R637 / ≠ Short MidRank MidBeta R631 / ≠ Short MidRank HiBeta R632 / ≠ Online / ≠ GRPO R583

**Host:** mine-r226-marsplan-fullft-1 (brave) GPUs **0,1** (R652 owns 2,3; R634 uplink live; R651 merge done)

**Decision rule:** Stage-5 iff fresh n80 paired margin > max(2·SE, 0.002) AND median |z|≥80 AND B pass≥0.30 vs **reign34** (v4 stamp k=3).
