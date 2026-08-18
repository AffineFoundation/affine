# R726 — TRAIN (p3766)

After R717 SoftCtx SuperExtra REFUTE ~−0.28×: move LoBeta MidRank onto **MidCtx** SuperExtra.

| knob | value |
|---|---|
| base | `unconst/Affine-5czsc2fc98-r252-merged`@`b42d6245…` |
| method | Offline-DPO · HiAlpha · MidRank · LoBeta · MidCtx · SuperExtra · ep3 · LoLR |
| β / r / α / lr | 0.02 / 32 / 128 / 1e-6 |
| max_len / steps | 8192 / 14400 |
| data | MidCtx pairs from R692 (604 lines) |
| GPUs | crown 4,5 · pid **111742** · wait→merge armed |
| decision | Stage-5 iff n80 margin > max(2·SE, 0.002) ∧ thought≥80 ∧ B≥0.30 vs **reign35** (v4 k=3) |

≠ SoftCtx SuperExtra R717 / ≠ ShortCtx SuperExtra R724 / ≠ UltraExtra MidCtx R692 / ≠ marsplan MidCtx R721.
