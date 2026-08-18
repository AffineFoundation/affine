# R726 — N80 LIVE (p3772)

After R717 SoftCtx SuperExtra REFUTE ~−0.28×: move LoBeta MidRank onto **MidCtx** SuperExtra.

| knob | value |
|---|---|
| base | `unconst/Affine-5czsc2fc98-r252-merged`@`b42d6245…` |
| method | Offline-DPO · HiAlpha · MidRank · LoBeta · MidCtx · SuperExtra · ep3 · LoLR |
| β / r / α / lr | 0.02 / 32 / 128 / 1e-6 |
| max_len / steps | 8192 / 14400 |
| data | MidCtx pairs from R692 (604 lines) |
| status | **MERGE_DONE** `/tmp/r726_merged` 66G/16sh · **N80 LIVE** crown **4,5**/:8002 vs reign35 · vllm**117256** sim**119974** · `*_reign35_wvk7` fail-closed k=3 · Triton seed chall_r717 n_star=30 |
| decision | Stage-5 iff n80 margin > max(2·SE, 0.002) ∧ thought≥80 ∧ B≥0.30 vs **reign35** (v4 k=3) |

≠ SoftCtx SuperExtra R717 / ≠ ShortCtx SuperExtra R724 / ≠ UltraExtra MidCtx R692 / ≠ marsplan MidCtx R721.

Check: `lium exec gentle-orbit-bd "tail -40 /root/logs/p3772_r726_chall_n80_wvk7.log; cat /root/affine_data/r726_decision_reign35_wvk7.json 2>/dev/null"`
