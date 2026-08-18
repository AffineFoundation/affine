# R727 — N80 LIVE (p3772)

After R718 SoftCtx SuperExtra REFUTE ~−0.40×: move HiRank LoBeta onto **MidCtx** SuperExtra.

| knob | value |
|---|---|
| base | `unconst/Affine-5czsc2fc98-r252-merged`@`b42d6245…` |
| method | Offline-DPO · HiAlpha · HiRank · LoBeta · MidCtx · SuperExtra · ep3 · LoLR |
| β / r / α / lr | 0.02 / 64 / 128 / 1e-6 |
| max_len / steps | 8192 / 14400 |
| data | MidCtx pairs (604 lines) |
| status | train.done 01:48Z → **MERGE_DONE** `/tmp/r727_merged` · **N80 LIVE** crown **6,7**/:8003 vs reign35 · vllm**120684** sim**122955** · `*_reign35_wvk7` · Triton seed chall_r726 n_star=30 |
| decision | Stage-5 iff n80 margin > max(2·SE, 0.002) ∧ thought≥80 ∧ B≥0.30 vs **reign35** (v4 k=3) |

≠ SoftCtx SuperExtra R718 / ≠ MidRank MidCtx LoBeta R726 / ≠ Soft HiRank SoftCtx R715/R716.

Check: `lium exec gentle-orbit-bd "tail -40 /root/logs/p3772_r727_chall_n80_wvk7.log; cat /root/affine_data/r727_decision_reign35_wvk7.json 2>/dev/null"`
