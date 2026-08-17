# R680 — r252 × Offline-DPO × HiAlpha × MidRank × HiBeta × LongCtx × UltraExtraSteps × ep3 × LoLR

**Axis:** Long MidRank HiBeta UltraExtra (max_steps=7200) after R668 Mega MERGE_DONE.
Amplify R604 ~0.26× / R668 Mega ep3 — sibling of R674 Long MidRank LoBeta UltraExtra.
≠ R668 Mega 3600 / ≠ R674 Long MidRank LoBeta UltraExtra / ≠ R678 Long HiRank MidBeta UltraExtra / ≠ R677 Long MidRank MidBeta UltraExtra / ≠ Online / ≠ GRPO.

| knob | value |
|---|---|
| base | `unconst/Affine-5czsc2fc98-r252-merged` @ `b42d6245…` |
| method | Offline-DPO on duel Reason pairs |
| β | 0.3 (HiBeta) |
| LoRA r/α | 32 / 128 |
| lr | 1e-6 (LoLR) |
| max_len | 16384 (LongCtx) |
| max_steps | **7200** (UltraExtra; Mega=3600) |
| epochs | 3 |
| GPUs | zesty 6,7 |
| decision | Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign34 (k=3) |
