# R1204 — Mega after Hyper REFUTE

**Status (p4331):** MERGE_DONE idle unblocked → chall:8004 + **v4 n80 LIVE** on mine-r340 GPU6.

| field | value |
|---|---|
| parent | R1182 MidCtx HiRank Midβ Hyper MidLR REFUTE m=+0.000328 SE=0.000775 ~0.16× thought✓175 B✓0.425; MidCtx HiRank Midβ Hyper LR exhausted (Ultra R1156 / Mid R1182 / Hi R1100) → ShortCtx HiRank Midβ Mega MidLR isolate |
| knobs | β=**0.1** α=128 r=**64** lr=**1e-6** @6144 Mega **max_steps=19200** |
| decision | Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36 |
| ops | wait script had **wrong EXP dirname** → MERGE_DONE sat idle; p4331 fixed path + lean_chall; n80 pid**106466** · chall pid**104786** :8004 · out `/root/affine_data/r1204_sim_result_reign36_wvk7.json` |
| poll | `tail -f /root/logs/p4316_r1204_chall_n80_wvk7.log` · SSH `18.118.83.97:40127` |
