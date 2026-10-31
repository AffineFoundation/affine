# SN120 trainer host v8.3.3i: direction step + support renorm (live from epoch 354)

Approved by Jacob 2026-10-31 07:00 CR after Allan confirmed the formula. One change on top of v8.3.3h (fp32 head master, kept on):

- **Direction step** (`sgd_direction: normgrad`): every tensor keeps the norm of the capped fresh-AdamW step
  (lr 5e-7, 2e-5 per-tensor relative cap, global cap unchanged, computed from the fp32 master), direction g_t/||g_t||:
  `Delta_t = fp32(s * r_t * ||d_t|| / ||g_t||) * g_t`, `M_C = M_H - Delta_t`, `W_C = bf16_rne(M_C)`. 1x, one step per turn.
- **Support renorm** (`support_renorm: {top_k 20, top_p 0.95, temperature 1.0}`, equal to the signed sampling settings):
  Allan's `patch/support_renorm.diff` (catoneone/sn120-k1-fixes@230b111) applied verbatim (the v83_dp hunk ported by hand,
  same changed lines). An out-of-support sampled token leaves the PG term only; KL and derail detection are unchanged.
  Files are never rejected; the per-turn outside fraction is published (training result `support_renorm`, training.json, turn.json).

| File | What |
| --- | --- |
| `host-v833i/` | the four changed live host files + signed manifest |
| `patch_launcher.py`, `launcher-v833h-to-v833i.diff`, `renorm-applied-v833h-to-v833i.diff` | how v833i was built from v833h |
| `sign_v833i.py` | ROOT signing (reads the seed path from the active lifecycle config; never prints it) |
| `tests/test_v833i.py` | sign direction bit-identical to v833h; normgrad equals Allan's math (max diff 0); support log q vs a reference |
| `activation/activate_rs.sh` | `MODE=stage\|switch\|revert`; switch only at a decision boundary (after DECISION, before the next job) |
| `activation/watchdog_ff.py` | `rs_check`: auto-revert on job failure/timeout, non-finite values, update norm outside [0.5, 2] x 0.0224 or master != raw, 8-vs-1 mismatch |
| `rs_8v1.py`, `pod/rs_replay_pod.py`, `pod/rs_compare_pod.py` | 8-GPU turn vs 1-GPU replay of the same batch (Allan's bf16_exp stream) |
| `patch_sidecars.py`, `services/` | training.json / mining.json / turn.json fields |
| `docs_patch.py` | llms.txt, step-recipe.json, k1/replay |

Revert: `MODE=revert bash activation/activate_rs.sh` (restores the v833h drop-in and worker config from the backup).
