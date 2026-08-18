# R727 — TRAIN (p3767)

After R718 SoftCtx SuperExtra REFUTE ~−0.40×: move HiRank LoBeta onto **MidCtx** SuperExtra.

| knob | value |
|---|---|
| base | `unconst/Affine-5czsc2fc98-r252-merged`@`b42d6245…` |
| method | Offline-DPO · HiAlpha · HiRank · LoBeta · MidCtx · SuperExtra · ep3 · LoLR |
| β / r / α / lr | 0.02 / 64 / 128 / 1e-6 |
| max_len / steps | 8192 / 14400 |
| data | MidCtx HiRank LoBeta pairs (R655/R618 lineage; 604 lines) |
| GPUs | crown 6,7 · pid **113050** · wait→merge armed |
| decision | Stage-5 iff n80 margin > max(2·SE, 0.002) ∧ thought≥80 ∧ B≥0.30 vs **reign35** (v4 k=3) |

≠ SoftCtx SuperExtra R718 / ≠ MidRank MidCtx LoBeta R726 / ≠ Soft HiRank MidBeta SoftCtx R715.
