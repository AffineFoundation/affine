# Isolated long-context role executor

`subnet.long_context_backend_jobs` is a prospective, separate executor for the
qualified 32K selective-head service runtime. It does not modify the active
GPU worker or introduce a blockchain writer. Actual deployment and common
native epoch qualification remain separate gates.

Invoke it in a fresh isolated source process:

```sh
python -m subnet.long_context_backend_jobs job.json --authority APPROVED_PUBLIC_KEY \
  --workspace /root/isolated-job-state --checkpoint-cache /root/approved-checkpoint
```

The signed operator job must contain the signed epoch manifest, exact checkpoint
file hashes, source closure, interpreter hash, torch/transformers/TOPLOC/NumPy
versions, role, expiration, and per-object direct R2 capabilities. The executor
checks signature authority before importing model or native factory code,
reading checkpoints, or downloading artifacts. The CLI imports only the fixed
`subnet.long_context_native_factory.create` deployment factory; jobs cannot
select an executable factory or supply private task databases. The deployment
factory is operator-owned trusted code in the approved source closure.

Both manifest and job bind `artifact_policy` to compressed 250,000,000 bytes
and raw 500,000,000 bytes. The manifest retains `transport_policy=direct-r2-v1`.
The independent ZIP reader preserves the existing bounded float32 NPY framing,
512 output rows, 200,000 vocabulary columns, and no executable serialization.
No array quantization or token/context truncation is introduced.

All model roles require the strict common SDPA-flash-only BF16 sm86 profile,
32,768 context limit, output-prediction-row full-vocabulary head, no TF32, two
threads, and unchanged zero-error TOPLOC / log-probability atol 1e-5, rtol zero.
Signed resource policy requires 12 GiB free memory for inference roles and
20 GiB for full training, with a bounded 1,800-second wait preserving other
jobs. Full training also enforces its separate measured 8 GiB allocator cap.

Mining cumulatively uploads complete K/L batches within the signed epoch
window. Verification performs full inference and fresh native replay for each
accepted batch. Training re-audits frozen submissions before applying the
separately signed full sequential agent-only optimizer policy for exactly one
selected audited pair and one step, exports the
new checkpoint and exact file map, and leaves six-object publication to a
later scoped operator-authorized upload. Evaluation records exact requested
held-out indices/seeds, task hashes, raw rollout/probability artifacts, and
explicit failures. Reports bind local artifact file hashes and sizes.

Thirteen unit controls cover signature-before-access, exact numerical and transport
policies, forbidden chain roles, full optimizer/resource policy, deadline
checks, strict JSON policy types, rejection of multi-step training before any
artifact access, float32 transport roundtrip, duplicate ZIP entries, and unbounded NPY
header rejection. These controls are not a claim that the common EOG service
has completed an epoch.

Multi-step training is rejected during signature/policy validation. The executor
does not reset Adam or reference probabilities across advertised steps. Its
current admission scope is `qualified-single-pair-single-step-v1`; reports list
the number of audited pairs separately from the one optimized pair. Persistent
multi-step training requires separate qualification before this restriction
can change. The qualified full sequential helper and frozen deployed worker
sources remain unchanged.
