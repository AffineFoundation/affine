# R709 result

## Status (p3756)
**MERGE_DONE → CHALL→N80 LIVE** on zesty GPUs **4,5**/:8002 vs reign34 (wvk=7 k=3 τ=0.03).

- merge: `/tmp/r709_merged` 16sh+visual (~66G) local — no SCP
- chall vllm pidfile `/root/logs/vllm_chall_r709.pid` · outer `/root/logs/p3756_r709_outer.pid`
- sim out: `/root/affine_data/r709_sim_result_reign34_wvk7.json` (pending)
- decision: `/root/affine_data/r709_decision_reign34_wvk7.json` (pending)
- axis: marsplan Soft MidRank LoBeta SoftCtx UltraExtra (β=0.02 r=32 @12288 steps=7200 ep=3 LoLR; transfer R637 ~1.45×)

## Decision rule
Stage-5 iff paired margin > max(2·SE, 0.002) **and** median |z|≥80 **and** B≥0.30.
