# R738 MERGE→N80 launch — p3790

- **When:** 2026-08-18T03:59:10Z
- **Why:** `r738_merge.done` @03:42Z; GPUs 6,7 free; no n80 armed (same failure mode as p3786 R739).
- **Where:** lunar-wolf-be (`mine-r165-awesome-hialpha-1`) GPUs **6,7** chall **:8003**
- **Chall:** pid **706677** `vllm serve /tmp/r738_merged` (16 shards)
- **King:** reign35 `tammyfritz/Affine-5hmwhnfbix-tammy2`@`7e5fd5f8…` on :8001
- **Sibling:** R741 TRAIN kept on 4,5
- **Check:** `tail -40 /root/logs/p3790_r738_chall_n80_wvk7.log`; progress `/root/affine_data/r738_sim_progress_reign35_wvk7.json`; decision `/root/affine_data/r738_decision_reign35_wvk7.json`
- **Gate:** Stage-5 iff margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 (v4 k=3)
