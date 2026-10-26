# Private fixed-token numerical qualification v3

This successor preserves the failed v2 package and its original evidence. V2
stopped before any forward pass because Transformers5.14's `rope_scaling` alias
returns the ordinary normalized `rope_parameters` dictionary rather than None.
V3 requires the exact original default mapping (`rope_type=default`,
`rope_theta=10000`), rejects added scaling or changed native context, and preserves
that mapping while changing context4096→8192 exactly as the original collector.
The four input groups and six diagnostic profiles are unchanged.

This is a separate, no-update diagnostic. It neither changes nor resumes the
original GRPO/preference experiment. The caller must supply a ROOT-signed scope
on a separate idle H200. There is no sampling, grader, optimizer, model export,
production client, tolerance pass, or held-out data in this worker.

Run after independent review and ROOT authorization:

```sh
python -m research.group_numerical_qualification_v3.worker --scope ROOT-SIGNED.json
```

The scope version is `private-group-numerical-qualification-v3`. It binds
`workspace`, `hostname`, `boot_id`, `gpu_uuid`, `created_at`, `expires_at`,
`duration_seconds`, `disk_reserve_bytes`, `files`, `source_files`,
`checkpoint`, and `runtime_versions`. Its authority and upstream checkpoint
come from the pinned original private-study worker. Every imported source is
staged in `workspace/source`; the launcher must pin this package plus the
reused fixed-group package and `subnet.backend_jobs` dependency closure.

`files` includes `INPUTS.json`, which points to the original signed scientific
plan, the signed capture receipt, and four byte-identical original group files.
The worker authenticates both envelopes and requires their order and hashes to
agree with `INPUTS.json` and the execution scope. The scientific plan, rather
than observed outcomes, fixes the first four complete control/update-zero groups.
Each group keeps all eight original attempts, tokens, EOS/cap information,
outcomes and actual BF16 generation logprobs. Nothing is regraded or replaced.

For each of BF16 and FP32, one model is loaded at a time. FP32 is an explicit
lossless widening of the same upstream BF16 parameter values. The BF16 digest
must match the original group stamp, and each dtype's digest must stay unchanged
across all its calls. The BF16 profile preserves the original FP32 buffer dtypes
(including RoPE inverse frequencies); buffers are inventoried and hashed before
and after each dtype's calls. Each model runs three profiles:

1. Forced-token cached replay with the original batch-eight shape, masks and
   positions. It uses the submitted token at every step, including EOS, without
   drawing another token. Ended rows use the original masked-EOS padding.
2. Original full-sequence scorer in eval/no-grad mode.
3. Original full-sequence scorer in train/gradient-checkpointed mode. It builds
   the forward graph and discards it; it never calls backward or an optimizer.

All profiles use T0.7 and vocabulary support151665. Original prompts and token
paths are unchanged; horizon4096/context8192 match the scientific plan. Selected
token LP vectors are written before comparison. Comparisons retain signed
per-token logratio vectors, mean/p99/max differences, PPO clip fractions for
[0.8,1.2], original EOS/cap strata, and per-profile timing/memory. FP32 values are
likelihoods of forced original BF16 trajectories, not evidence of FP32 sampling.
The report declares no new numerical admission threshold.

`numerical-output/` must not already exist. It contains original scope, immutable
profile/comparison JSON files, per-dtype parameter digests, an observation log,
and `RESULT.json` only if all requested profiles finish. It contains no model or
optimizer snapshot. The supervisor owns runtime limits, failure capture, and
eventual rental termination. A failure must be captured without an automatic
retry or omission of its evidence.

Fifteen tiny CPU tests exercise causal token alignment, batch padding, early EOS,
budget stops, duplicate text at different attempts, train/eval/gradient modes,
unchanged parameters, buffer dtype preservation, actual Transformers5.14.1
configuration normalization without model construction, deadline hooks and asymmetric PPO clipping:

```sh
.venv/bin/python -B -m unittest research.group_numerical_qualification_v3.test_profiles -v
```

These tests do not qualify real H200 numerical behavior. No model has been run
by this package's implementation work.
