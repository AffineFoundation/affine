# R797 plan — marsplan MidCtx MidRank HiBeta Mega UltraLoLR

After **R787 REFUTE** (~−0.80× Soft MidRank HiBeta Soft UltraLoLR) free zesty 6,7.
Axis: **R766** MidCtx MidRank HiBeta Mega @1e-6 (~−0.30×) → UltraLoLR sibling.

- base: `marsplan0624/affine-5gedzafcvg-queen`@`556d02a2`
- β=0.3 · α=128 · r=32 · max_len=8192 · max_steps=19200 · ep=4 · lr=5e-7
- pod: zesty-comet-da GPUs **6,7** (R796 TRAIN on 4,5)
- ≠ R766 @1e-6 / ≠ MidCtx Mid Mid UltraLoLR R796 / ≠ Soft Mid Hi Soft UltraLoLR R787
- decision: Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign35
