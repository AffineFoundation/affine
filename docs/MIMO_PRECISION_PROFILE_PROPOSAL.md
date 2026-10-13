# MiMo precision profile: prospective, not activated

This change adds `qwen3_5-mimo-precision-v2`. Historical `qwen3_5-mimo-v1` parsing, serialization and default creation remain unchanged. No new source archive, signed release, manifest, service or chain transaction is created by this patch.

## Why

Independent replay reproduced four sampling mismatches after native grading, selected T1 probabilities and TOPLOC passed. A bounded 20-case experiment on one spare H200 demonstrated that enabling TF32 or medium float32 precision changes two disputed token boundaries. Those settings also create earlier mismatches, so they do not explain or validate complete submitted trajectories. This is a demonstrated compatibility sensitivity, not proof of miner intent and not justification for widening CDF tolerances.

## Contract and implementation

A new runtime-v2 spec requires `precision_profile` with version `mimo-qualified-precision-v1`. Its digest participates in the existing signed runtime and draw binding. The only initially supported profile preserves the measured reference: matmul TF32 disabled, highest float32 precision, cuDNN TF32 enabled, BF16/FP16 reduced-precision reductions enabled, deterministic algorithms disabled. Parameter/recurrent dtypes, attention and sampler remain separately pinned as before.

The runtime configures this profile before model construction/cache creation. Existing inference, prefill, scoring and checkpointed backward recomputation paths guard against ambient autocast and changed global flags. Forward guards fail; they do not silently reset flags mid-cache. `PrecisionMismatch` is a RuntimeError and propagates outside the worker's InvalidSample handling, so an execution compatibility failure cannot be turned into a fabricated invalid miner audit.

New v2 worker reports include locally observed execution provenance covered by the normal worker report signature. Host report admission validates its profile digest, flags, device and bounded build/hardware fields. It observes the trusted worker process, not an external miner's actual execution and not every subprocess via hardware attestation. GPU/driver/build/thread provenance is diagnostic metadata, not a task identity or reward input. A missing driver observation is null, never guessed. Original v1 reports cannot acquire synthetic v2 provenance.

The new revision is recognized by backend dispatch, the CLI and miner supervisor. Older clients reject it; operators must publish and qualify the updated client/source before activating it. The default `make_runtime_spec` still creates v1. Opt into v2 explicitly using `precision_profile=profile_dict()` after qualification.

## Required GPU qualification before release

Use a fresh ROOT-signed nonpayable private candidate release/job binding this exact reviewed source and v2 profile. Do not modify original issued manifests or use unsigned operator policy. Preserve original scientific evidence and temporary deferral decisions independently.

1. Generate complete honest eight-nonce groups using the official B1 cached sampler, fresh cache per rollout, all nonce 0..7 trajectories retained. Cover ordinary tasks, long/capped trajectories and the five investigated task indices. Select groups for diagnostic coverage independently of pass/fail outcomes. Do not replace the eight original draws with cherry-picked attempts.
2. On spare retained GPU0 generate, then verify every rollout on GPU0 and GPU1. These are two devices on one host, **not two independent hosts**. Capture full distributions only as private bounded diagnostics where needed. Confirm native outcomes, selected T1 arrays, TOPLOC and cached fallback.
3. Before release, repeat on a genuinely separate qualified verifier host and obtain a representative external miner reproduction. Two devices on the spare host alone do not establish external compatibility. Record GPU, driver, package/source fingerprints, optional kernels, cache/attention geometry and launch flags.
4. Run token substitutions at early/middle/late and near-CDF-boundary positions, changed nonce/task/checkpoint/prompt, duplicate or omitted nonces, altered stopping metadata and proofs. Require rejection at the appropriate layer. Include honest repeated-output cases; duplicate text alone is not forged sampling.
5. Inject TF32, float32 precision, BF16-reduction, deterministic and ambient CPU/CUDA autocast changes. Require a compatibility failure before acceptance/invalid penalty evidence, including during backward recomputation. Confirm no optimizer step or checkpoint publication after a guard failure.
6. Run one private GRPO update from honest groups and evaluate it with the same pinned runtime. Check finite losses/gradients, unchanged data selection/objective settings and model-only publication. This is an integration check, not evidence of convergence.
7. Compare false rejections and timings against the qualified baseline. Preserve all original diagnostic/report hashes. Review the new source and client compatibility documentation, then activate only at a new epoch boundary. Continue historical audits with their original contract. No blanket tolerance relaxation.

## Validation scope

Unit tests cover strict profile schema, byte-preserved v1, each numeric flag, both autocast devices, pre/post-forward guards and bounded provenance. Real tiny HF hybrid CPU tests cover cached decode, prefill, gradient-checkpointed training and rejection when flags change before backward. Worker integration tests ensure compatibility errors do not become invalid-sample reports and v2 report provenance is required and bound. These tests do not substitute for the full GPU qualification above.

Selected regression suite: **81 tests and 18 subtests passed** (precision, runtime, worker, GRPO worker, miner supervisor). No GPU v2 qualification has run yet. The retained spare and old trainer pod share a Lium executor and cannot establish cross-host compatibility; a distinct idle executor must be used for that stage.
