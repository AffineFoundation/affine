# R674 — r252 × Offline-DPO × HiAlpha × MidRank × LoBeta × LongCtx × UltraExtraSteps × ep3 × LoLR

**Axis:** Long MidRank LoBeta UltraExtra (max_steps=7200) after R662 Mega MERGE_DONE.
Amplify R643 ~0.50× / R579 ~0.46× / R662 Mega ep3 — sibling of R673 Soft MidRank LoBeta UltraExtra.
≠ R662 Mega 3600 / ≠ R673 SoftCtx UltraExtra / ≠ R664 MidBeta / ≠ R668 HiBeta Long / ≠ Online / ≠ GRPO.

| knob | value |
|---|---|
| base | `unconst/Affine-5czsc2fc98-r252-merged` @ `b42d6245…` |
| method | Offline-DPO on duel Reason pairs |
| β | 0.02 (LoBeta) |
| LoRA r/α | 32 / 128 |
| lr | 1e-6 (LoLR) |
| max_len | 16384 (LongCtx) |
| max_steps | **7200** (UltraExtra; Mega=3600) |
| epochs | 3 |
| GPUs | zesty 4,5 |
| decision | Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign34 (k=3) |
