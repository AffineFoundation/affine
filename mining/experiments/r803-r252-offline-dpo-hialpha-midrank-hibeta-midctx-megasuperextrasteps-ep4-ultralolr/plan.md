# R803 — r252 MidCtx Mid Hi UltraLoLR
Pod: golden (R262) GPUs 6,7 (after R791 REFUTE). Local n80 after merge on :8003 (R792 keeps :8002 on 4,5).
Base: unconst/Affine-5czsc2fc98-r252-merged@b42d6245
β=0.3 α=128 r=32 lr=5e-7 @8192 max_steps=19200 ep=4
Parent signal: R791 MidCtx MidRank MidBeta Mega UltraLoLR REFUTE ~−1.32× → HiBeta sibling.
≠ R791 Mid Mid MidCtx UltraLoLR / ≠ R792 Mid Lo MidCtx UltraLoLR / ≠ Soft Mid Hi Soft UltraLoLR / ≠ Online / ≠ GRPO
Decision: Stage-5 iff v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign35.
