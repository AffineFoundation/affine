# R788 plan — tammy Soft HiRank MidBeta SoftCtx Mega UltraLoLR

After **R776 REFUTE** (~−0.28× @ lr=1e-6), same SoftCtx HiRank MidBeta Mega recipe with **lr=5e-7**.

- base: `tammyfritz/Affine-5hmwhnfbix-tammy2`@`7e5fd5f8`
- β=0.1 · α=128 · r=64 · max_len=12288 · max_steps=19200 · ep=4
- pod: gentle-orbit-bd (crown) GPUs **6,7** (R786 on 4,5)
- decision: Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign35
