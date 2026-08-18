# R787 plan — marsplan Soft MidRank HiBeta SoftCtx Mega UltraLoLR

After **R777 REFUTE** (~−0.72× @ lr=1e-6), same SoftCtx MidRank HiBeta Mega recipe with **lr=5e-7**.

- base: `marsplan0624/affine-5gedzafcvg-queen`@`556d02a2`
- β=0.3 · α=128 · r=32 · max_len=12288 · max_steps=19200 · ep=4
- pod: zesty-comet-da GPUs **6,7** (R785 on 4,5)
- decision: Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign35
