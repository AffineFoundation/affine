# R937 — TRAIN launched (p4042)

- Axis: vera Offline-DPO HiAlpha HiRank Midβ SoftCtx Soft Mid Mid Soft UltraLoLR
- Knobs: β=0.1 α=128 r=64 lr=5e-7 @12288 epochs=4 max_steps=19200
- Host: mine-r338 calm-fox-6a GPUs 4,5 (after R338 chall reap)
- TRAIN pid **24800** @2026-08-19T21:02:27Z; wait→merge armed pid **24805**
- Parent: R338 online-DPO REFUTE m=−0.003241 SE=0.004677 z=−0.693 n=78 ~−0.35× thought✓218 B✓0.399 k=3/τ=0.03
- Isolate: SoftCtx HiRank Midβ ≠ SoftCtx Loβ R936 / ≠ SoftCtx MidLoβ R931 / ≠ SoftCtx Hiβ R862 / ≠ MidCtx Midβ R928 / ≠ MidCtx Loβ R935
- Decision: Stage-5 iff fresh v4 n80 margin>max(2·SE,0.002) AND thought≥80 AND B≥0.30 vs reign36

