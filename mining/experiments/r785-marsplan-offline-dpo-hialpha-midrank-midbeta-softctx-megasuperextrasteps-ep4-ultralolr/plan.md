# R785 plan — marsplan Soft MidRank MidBeta SoftCtx Mega UltraLoLR

After **R772 REFUTE** (~−0.57× @ lr=1e-6), same SoftCtx MidRank MidBeta Mega recipe with **lr=5e-7**.

- base: `marsplan0624/affine-5gedzafcvg-queen`@`556d02a2`
- β=0.1 · α=128 · r=32 · max_len=12288 · max_steps=19200 · ep=4
- pod: zesty-comet-da GPUs **4,5** (R777 on 6,7)
- decision: Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign35
