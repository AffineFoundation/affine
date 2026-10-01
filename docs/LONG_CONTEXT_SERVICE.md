# Prospective long-context service runtime

`subnet.long_context_service_runtime.LongContextServiceRuntime` exposes configure,
for_environment, rollout, verify, and full_parameter_train for an isolated future
EOG service. It requires an operator-injected trusted native session factory;
miner artifacts cannot supply executable factories. Existing live runtime and
controller profiles are unchanged. Common controller deployment remains a
separate integration gate.

The runtime keeps complete model context up to 32,768 tokens and computes the
full vocabulary only at output-prediction rows. Every emitted token retains its
full float32 log-probability row and TOPLOC evidence. The same-backend BF16 CUDA
sm86 profile uses explicit SDPA flash attention, deterministic execution, two
threads, no TF32, and no cache. Verification keeps log-probability atol 1e-5,
rtol zero, and zero TOPLOC errors. Candidate likelihood reduction is explicitly
bound to NumPy float32 summation in the service profile.

`subnet.long_context_service_training` defines a separate full-parameter,
sequential-agent-turn mean-log-probability DPO policy. Reference probabilities
are computed before updates. The scalar loss derivative is then accumulated
through one complete-context turn graph at a time before a single optimizer
step. All agent output tokens contribute; auxiliary model-role outputs cannot
be used as agent training targets. Parameters, gradients, and Adam moments are
BF16, without FP32 master weights. Gradient checkpointing is explicit.

## Actual isolated qualification

The signed job `state/long-context-eog/sequential-seven-optimizer-job.json`
consumed independently admitted seven-turn original EOG positive and
first-patch-only negative controls. Both contained 300 agent tokens. The
494,032,768-parameter model produced gradients for all 290 parameter tensors;
252 tensors and 222,642,042 parameter elements changed after one step at
learning rate 5e-5 and beta 0.1. Measured peak GPU allocation was
4,602,542,592 bytes under the separately signed 8 GiB allocator cap and
20 GiB free-memory admission guard. The fitted pair margin increased by
0.06636; this does not establish held-out quality improvement.

The successor checkpoint is
`037dbfb51c8ad9cda68f098f8e1a5ff0a345d796532cb94439f4745db5c3f979`.
Only six object-specific presigned PUT capabilities went to the retained GPU
worker. The operator independently streamed every published checkpoint file
and checked its complete hash and size before publishing its signed descriptor.
No bucket account credentials or signing keys went to the worker.

A separate fresh successor run recomputed and independently verified all seven
positive model turns, including the terminal DONE token. Fresh original native
replay and grading returned reward 1.0. The signed completion receipt is
`state/long-context-eog/full-sequential-seven-1790849232/seven-turn-completion.json`;
raw successor evidence and native admission are in
`state/long-context-eog/eog-post-sequential-positive-1790849794`.

These are curated target-model computation and optimizer controls, not claims
of original sampling, a common production epoch, or generalized improvement.
No blockchain weights or transactions were submitted.
