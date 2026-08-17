# R703 — marsplan Soft MidRank HiBeta SoftCtx HyperExtra ep3×LoLR

## Status (p3745)
**CHALL→N80 LIVE** on zesty GPUs **4,5**/:8002 (local MERGE_DONE 16sh/66G; no SCP).

- vllm pid **836892** CHALL_READY @ 2026-08-17T22:15:34Z
- sim pid **839149** (`*_wvk7` fail-closed k=3)
- Triton seed `chall_r703` from `chall_r702` n_star=26
- Out: `/root/affine_data/r703_sim_result_reign34_wvk7.json`
- Progress: `/root/affine_data/r703_sim_progress_reign34_wvk7.json`
- Decision: `/root/affine_data/r703_decision_reign34_wvk7.json`
- Log: `/root/logs/p3745_r703_chall_n80_wvk7.log`

## Axis
marsplan Soft MidRank **HiBeta** SoftCtx HyperExtra (β=0.3 r=32 @12288 steps=10800) — completes Soft MidRank SoftCtx HyperExtra β sweep vs R701 Lo / R702 Mid.

## Check
`lium exec zesty-comet-da 'tail -40 /root/logs/p3745_r703_chall_n80_wvk7.log; cat /root/affine_data/r703_decision_reign34_wvk7.json 2>/dev/null; cat /root/affine_data/r703_sim_progress_reign34_wvk7.json'`
