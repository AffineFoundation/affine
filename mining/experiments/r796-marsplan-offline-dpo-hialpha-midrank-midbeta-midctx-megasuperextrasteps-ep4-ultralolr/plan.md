# R796 — marsplan MidCtx MidRank MidBeta Mega UltraLoLR
- base: marsplan0624/affine-5gedzafcvg-queen@556d02a2
- Offline-DPO MidCtx Mega: β=0.1 α=128 r=32 lr=5e-7 @8192 max_steps=19200 ep=4
- Parent: R764 MidCtx Mid Mid Mega @1e-6 near-miss ~0.12× → UltraLoLR; R785 Soft Mid Mid Soft UltraLoLR REFUTE ~−0.17× → MidCtx transfer
- ≠ R764 @1e-6; ≠ Soft Mid Mid Soft UltraLoLR R785; ≠ r252 MidCtx Mid Mid UltraLoLR R791; ≠ Online / ≠ GRPO
- Decision: Stage-5 iff v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30
