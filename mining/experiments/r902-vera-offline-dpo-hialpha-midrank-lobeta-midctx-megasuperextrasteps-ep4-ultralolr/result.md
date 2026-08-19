# R902 — vera MidCtx Loβ UltraLoLR

**Status:** n80 LOADING p4015 (MERGE sat idle after train→merge waiter only)
**Axis:** Soft Mid Mid Soft MidRank LoBeta MidCtx UltraLoLR (β=0.02 r=32 α=128 lr=5e-7 @8192 ep4 steps=19200)
**Parent signal:** R885 SoftCtx Loβ ~−0.05× + R886 SoftCtx MidLoβ ~0.27× → MidCtx Loβ transfer
**Pod:** mine-crown-1 GPUs 4,5 :8002 · outer pid 401802 · merge `/tmp/r902_merged` SIZE_OK
**Decision rule:** Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36
**Check:** `tail -f /root/logs/p4015_r902_chall_n80_wvk7.log` · result `/root/affine_data/r902_sim_result_reign36_wvk7.json`
