# R693 result

**Axis:** Soft HiRank HiBeta SoftCtx UltraExtra (β=0.3 r=64 @12288 steps=7200 ep=3 LoLR) on `unconst/Affine-5czsc2fc98-r252-merged`@`b42d6245`.

**Decision rule (pre-registered):** Stage-5 iff fresh v4 n80 paired margin > max(2·SE, δ=0.002) **and** median |z|≥80 **and** B pass≥0.30 vs reign34 (k=3, τ=0.03).

**Status (p3751):** **N80 LIVE** on `mine-crown-1` GPUs **4,5**/:8002 — local `/tmp/r693_merged` 16sh/66G (no SCP); vllm**90524** loading (Triton seed chall_r705 n_star=30); outer**90361**; sim `*_wvk7` fail-closed k=3. First screen of this MERGE (p3737 skipped n80 for R705 HyperExtra train — later REFUTE −0.41×).

**Pending:** harvest `r693_sim_result_reign34_wvk7.json` → decide submit / REFUTE / next axis.
