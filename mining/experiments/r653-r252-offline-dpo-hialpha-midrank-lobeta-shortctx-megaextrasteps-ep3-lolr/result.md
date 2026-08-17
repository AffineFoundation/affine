# R653 — CHALL loading (p3697)

**Status:** SCP_READY 16:57:55Z → lean launched → vLLM :8003 TP2 loading (pid429790).
**Axis:** Short MidRank LoBeta ShortCtx Mega ep3×LoLR (β=0.02 r=32 lr=1e-6 @6144).
**Merge:** `/tmp/r653_merged` on golden — 16 shards / 66981 MiB; all shard byte sizes match brave source.
**v4 gate:** outs `*_reign34_wvk7.json`; fail-closed if `duel_params.n_teacher_samples ≠ 3`.
**Check:** `curl :8003/v1/models`; `tail /root/logs/p3693_r653_chall_n80_wvk7.log`; harvest when `r653_reign34_wvk7_pipeline.done`.
