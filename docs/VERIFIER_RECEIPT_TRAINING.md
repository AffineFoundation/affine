# Authenticated verifier receipts for training admission

The receipt implementation is sealed as source bundle `94ff74eb335e24d4702da2ec10cc0aee068076b003e6c0b13b0e81f5090bc79c` and independently read back from R2. The E9 controller now runs that source, and the large H200 training endpoint passed source/checkpoint admission. E9 training remains held until its final accepted receipt inventory is authorized; no successful receipt-admitted GPU training run is claimed yet. The controller is restricted to finishing E9, preventing an automatic next epoch with mismatched role sources.

The independent verifier performs the expensive model computation, probability, TOPLOC, exact sampling replay and environment checks. The trainer **does not repeat any of those checks**. It authenticates an operator-signed receipt, downloads the exact original frozen ZIP, checks its SHA256, byte length and bounded non-executable structure, and extracts only the batch and success/failure rollout digests accepted by that receipt. Training forward/backward passes and reference-model probability calculations remain training work.

The admission version is `authenticated-verifier-receipts-v1`, required by the covered-v3 and persistent-v4 training paths. Missing receipts fail closed. Older immutable scientific sources remain unchanged; this candidate does not reinterpret their historical training reports.

## Authentication chain

Before signing a compact admission, the coordinator reads the **original complete** verifier job row from its authoritative SQLite queue. It authenticates the operator-signed original job and manifest and the registered verifier's signed original report request. The stored job, worker, report and request hashes must agree. The report must bind the original job, epoch, checkpoint, source inventory, runtime, numerical profile, frozen ZIP and sampling assurance. An accepted batch must have an exact `fully_audited: true, valid: true` outcome and meet the original K/L quota. An unfinished row, unknown verifier or changed report cannot mint an admission.

The compact operator-signed payload commits:

- Version, epoch, checkpoint, original scientific computation binding and source SHA.
- Original signed manifest, signed job, job payload, worker report request and report SHA256 values.
- Registered verifier identity, original source/runtime inventory SHA256 values and original signed report timestamps.
- Miner identity, original frozen object key, ZIP SHA256 and exact compressed size.
- Each accepted batch's original slot, task identity and batch digest, plus the individual positive and negative rollout digests.
- `inference_TOPLOC_sampling_environment_verified_by: "registered-verifier"` and `trainer_verification_required: false`.

The operator and registered verifiers remain trusted admission authorities. A receipt authenticates their original accepted report; it is not a new cryptographic claim about historical physical execution. No miner-provided receipt or unsigned audit is accepted as operator authority.

## Current E9 execution amendment

E9's original signed miner challenge, sampler, task indices, checkpoint, harness, scoring and original verifier jobs remain immutable. Before its first training job, the operator must separately approve a private `private-training-execution-amendment-v1` envelope. Only the new controller and training endpoint use the new source bundle. Original e415 miner, verifier and evaluator endpoints keep their original metadata. New controller receipt checks apply only to covered-v3/persistent-v4 **train** reports, so old verifier reports do not need new receipt fields.

The private signed amendment contains exactly:

```json
{
  "version": "private-training-execution-amendment-v1",
  "epoch": "<original epoch>",
  "original_signed_manifest": "<original operator-signed public challenge envelope>",
  "original_signed_manifest_sha256": "<canonical envelope SHA256>",
  "training_source_bundle": "<approved new sealed source descriptor>",
  "training_policy": "bf16-full-adamw-covered-fixed-reference-v3",
  "training_input_policy": "authenticated-verifier-receipts-v1",
  "steps": 3,
  "verifier_receipt_inventory": [
    {
      "submission_sha256": "<original ZIP SHA256>",
      "verifier_receipt_sha256": "<canonical signed admission envelope SHA256>",
      "accepted_batch_sha256": ["<accepted batch SHA256>"]
    }
  ],
  "created_at": "<authorization UTC Unix timestamp>",
  "expires_at": "<first-request deadline, at most 24 hours later>"
}
```

This example shows the shape; it is not an admission artifact. Current E9 requests **three** optimizer updates. CP093's fifteen prior updates are historical context, not E9's requested step count. The actual source descriptor, original envelope, sorted receipt inventory and timestamps are required. The private training manifest changes the source descriptor and adds the input policy/amendment. Validation compares its original scientific fields against the authenticated original public challenge. Frozen audit/coverage fields are added normally after closure. The source must differ, while training policy, objective and requested step count remain original. Receipt issuance is deterministic from original accepted report timestamps, so unchanged recovery retains exact receipt signatures and hashes.

The controller's private `remote` configuration supplies:

```json
{
  "training_execution_amendment_required_epochs": ["<E9 epoch>"],
  "training_execution_amendment_files": {
    "<E9 epoch>": "<private approved signed amendment path>"
  }
}
```

The required-epoch gate runs before any capacity probe or training job signature/request. File presence only allows preparation to proceed; full amendment and receipt validation happens before the job is signed. A previously issued training request must recover its exact original amendment and admissions. A new amendment cannot replace an existing job. Authorization expiry controls the original first request; later recovery validates that original request's creation time, rather than authorizing another training execution.

## Reports and persistent state

Training reports contain `training_admissions` and an empty `audits` list. They state `trainer_verification_performed: false` and `all_pairs_authenticated_verifier_receipts: true`; the previous `all_pairs_independently_reaudited` claim is prohibited. The controller binds cached metrics to the exact original signed training request, receipt inventory, accepted batch hashes and report. Persistent-v4 retains the exact latest parent descriptor, namespace, optimizer counters, original job and independently read-back state publication rules. An unchanged BF16 inference hash still cannot reset or substitute optimizer state.

GPU qualification and live source activation remain operator-controlled steps. Local synthetic controls verify protocol integrity and absence of trainer verification calls; they do not establish GPU capacity, convergence, cross-hardware reproducibility or a completed production cutover.

## Source alignment at the next epoch boundary

The private E9 bridge authorizes E9 only. It does not permit E10 to publish the old e415 source descriptor while using new training source without another explicit amendment. Before the next epoch opens, the operator must coordinate all-role adoption of the tested sealed source: all verifier endpoints, miner, evaluator and trainer, plus prospective source/reward-writer approval. Existing E9 queue jobs, leases, worker report requests, signed opening and historical source archives retain their original bindings during that transition. If coordinated next-epoch source admission is incomplete, opening must remain held; there is no automatic old-source training fallback.
