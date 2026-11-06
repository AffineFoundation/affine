# Eval runtime audit — parallel cutover still in validation

These are the running gate's probe bits, not a claim that parallel evaluation is
live. The infrastructure cutover is blocked until its integration/recovery checks
pass. No gate threshold, sampling policy, reward rule or serving precision changes
are included here.

## Verify before registering

From this checkout:

```sh
python3 miner/kit/verify_eval_runtime.py
# Optional, inside your serving environment:
python3 miner/kit/verify_eval_runtime.py --check-packages
# Separately, inside the probe-worker environment:
python3 miner/kit/verify_eval_runtime.py --check-worker-packages
```

The CPU check verifies every file in `PINS.json`, the decompressed live task
snapshot, all 21,574 eligible task-to-probe-index mappings, bf16 serving, the
69,632 context limit and the 512-sequence default. It does not start a server,
install software, contact a service or reveal a private plan. This check has been
run; a complete GPU replay of this published package has not yet been certified.

`serve.sh`, `podctl.py` and `probe_worker.py` are in the parent directory. This
runtime directory adds the byte-identical 64K `complete_group_harness.py`, unchanged
`environments.py`, the full environment registry, and a compressed copy of the
live public gate-pool snapshot. The registry contains deployment paths; an eval
image places the snapshot at its configured `task_snapshot` path. Only public
code and tasks are included, never miner files, role credentials, task draws,
per-task priors or a private head baseline.

## Effective recipe

- Serving environment: vLLM **0.32.0**, torch distribution **2.13.0**
  (CUDA build `2.13.0+cu130`), transformers **5.19.0**.
- Separate probe-worker environment: torch **2.14.0**, transformers **5.14.1**.
  Both environments and seven runtime files were checked read-only on the live
  probe and both 8xH100 shadow nodes; they match exactly.
- bf16, prefix caching, GPU memory utilization 0.9, model length **69,632**,
  `MAX_SEQS=512`; no fp8 or quantized KV cache. Live parallel-eval nodes are
  intended to use 8xH100/H200/B200/B300. **No 5090 live-eval configuration.**
- Sample-count probe, not a time budget: expected Poisson draw **N=500** (the
  actual n is round-specific), **8** candidate samples and **12** head samples
  per task, output limit **65,536**. Effective plan max_tokens overrides the
  old manifest's default output length; do not use that old default.
- Temperature 1.0, top_p 0.95, top_k 20, thinking enabled; same public grader.
- Same `baseline_v1` estimator as the live prior-v1 gate. Its effective cfg is
  strict **z > 1**, positive gain, cap_max **0.33**, min_scored **100**, minimum
  draw coverage **0.95**. Legacy top-level 0.2/200 fields are not the prior-v1
  estimator's fields. This documents the actual code path; it changes nothing.

The snapshot has **21,578 indexed rows**, while the draw-eligible pool has
**21,574 tasks** (7,492 affine_math + 14,082 dapo). Four math indices
3322/6297/6512/6801 are excluded from the eligible pool. The other indices are
identical: `math:i -> i`, `dapo:j -> 7496+j`. Do not compress away those four
holes or substitute the retired v1 snapshot: that silently changes task mapping.

## Local simulation versus exact replay

Use `miner/kit/gate_eval.py` for an independent local draw and the public gate
math, with your own head/candidate serving endpoints and your own prior samples.
Its own draw is not the validator's hidden draw. Hardware sampling variation is
also possible even with the same settings.

To replay a particular decision exactly, the same checkpoint, round plan and
head baseline are needed, using the published request-seed rule and
`baseline_v1.cand_seed(round_seed, epoch)`. Follow the existing gate's delayed
public/reveal policy: **this publication does not promise that private priors or
baseline cells are available**. Do not claim a local z should equal the ledger
without the identical inputs. The task pool itself is public, not held out.

## Batch file version is a literal

The top-level batch `version` must be exactly **`affine-batch-v1`** or
**`affine-batch-v2`**. Do not put a trainer release, kit version, `v1` or `v2` in
that field. Check your file with `miner/kit/check_batch.py` before uploading.
