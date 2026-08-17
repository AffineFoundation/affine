# R687 Soft MidRank HiBeta SoftCtx UltraExtra — p3713

- **Axis:** r252 Offline-DPO HiAlpha MidRank HiBeta SoftCtx UltraExtraSteps ep3×LoLR
- **Knobs:** β=0.3 α=128 r=32 lr=1e-6 @12288 max_steps=7200 ep=3
- **Parent:** R651 Mega ~0.12× / R600 ~0.61×
- **Pod:** brave-raven-a9 GPUs **2,3** (idle after R679 MERGE; R681 SCP uplink ≠ train)
- **Launch:** 2026-08-17T18:50:49Z pid**231261** + wait→merge armed
- **Decision:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign34
