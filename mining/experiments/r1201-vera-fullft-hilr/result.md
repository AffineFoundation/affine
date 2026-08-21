# R1201 — result notes

## Status
- **p4321 REFUTE v4** vs reign36 · m=**−0.003071** SE=0.003308 z=−0.928 n=80 · bar≈0.006615 (~**−0.46×**)
  thought✓(med171) B✓(0.55) k=3/τ=0.03 · challenger_wins=false
  → FullFT MidLR (R1191) + HiLR (R1201) exhausted → **R1209** UltraLoLR same pass.
- **p4320:** teacher OOM@0.90 → serve_three util0.85/0.80 → n80 pid48168.
- **p4317:** After R1191 REFUTE ~0.02× → FullFT HiLR TRAIN.

## Axis
vera×FullFT×HiLR lr=2e-6 @8192 thought-only. ≠ R1191 MidLR · ≠ Offline-DPO · ≠ Online-DPO · ≠ GRPO.

## Decision
Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36 — **failed**.
