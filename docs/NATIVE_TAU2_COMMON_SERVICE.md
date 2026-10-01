# Controlled native Tau2 common service

This isolated coordinator opens a fresh nonpayable challenge for the chain-verified owned UID 131. It retains the qualified Qwen 0.5B agent and independently fixed SmolLM 135M auxiliary user, each in a separate remote framed worker. Original scenario-disjoint tasks, complete requests/tools, native tool state and grader remain operator controlled. General external miners are not supported.

The service generates new attempt 0/1 inside a signed 1200-second upload window. Both role-proof samples are privately uploaded through one encrypted, object-specific R2 PUT capability. The operator freezes an atomic GET, authenticates every role timestamp and exact archive byte, and runs a separate fresh full-model/native audit. Historical prerequisite samples receive no new epoch credit. Scores are proposed only; no blockchain weight submission exists in this module.

A matched agent decision drives one full-parameter AdamW mean-logprob preference update, recomputing the reference at the current checkpoint. Every auxiliary token is excluded from loss. Six new checkpoint objects receive specific PUT capabilities only after their digests exist; the operator independently streams their hashes. A successor native trajectory uses a newly published challenge. The fixed 16 scenario-disjoint heldouts use an independently signed autoregressive agent policy rather than the curated discovery policy. Failed/partial episodes are recorded separately from completed rewards with the requested 16-task denominator; improvement is not guaranteed.

Signing material and R2 account credentials stay on the operator. Disk headroom is checked before opening, every native job and every role request. Only this service's freshly audited raw caches may be removed after exact private-R2 upload, independent stream hash and a signed inventory; existing archives and controls remain untouched. Capacity failures are visible waiting states. Native process identity and remote PID/start-time journals prevent an observation failure from authorizing a duplicate model job. An unresolved remote process identity requires operator reconciliation.

Run the operator-only configuration with `python -m subnet.native_tau2_common_service --config state/native-tau2-common-live/config.json`, or install `systemd/affine-native-tau2-common.service` as a user unit. Logs, job identities, freeze receipts, scores, evaluation results and checkpoint publication records are private under that state directory. The service unit deliberately avoids automatic restart after an unresolved process identity.

The eight unit controls cover authority-before-artifact access, no payable jobs, disk admission, source links, auxiliary loss exclusion, private files and remote observation/retry semantics. They are software controls, not numerical or native episode qualifications. Actual live status must come from the private service journal and signed epoch artifacts.

The first source instance uses a separately authenticated final-boundary selector, `ops/finalize_native_tau2_common_boundary.py`. Its predeadline atomic receipt is admission evidence; after the published deadline the selector holds the exact coordinator identity, checks the final staging body and server modification time, rejects late/changed objects, then publishes an immutable final object and signed selection before resuming audits. The current source and challenge are preserved. A separately signed observer may retain terminal failed heldout arrays in private R2 as explicitly unadmitted storage evidence; it does not touch active/successful role checks.

The prospective `native_tau2_common_service_streaming.py` source adds independently pinned `native_tau2_common_role_storage.py` and `native_tau2_common_streaming_driver.py`. It uploads each new role array with exact-byte readback before removing its local cache, and records a signed inventory with `model_or_native_admission_claimed=false`. Fresh verification retrieves one full array at a time and still recomputes every model role and original native episode. The new source is not an approval to relabel historical receipts, and it must open a new signed challenge. Seven storage controls reject corrupt bodies, wrong sizes/prefixes and premature cache removal; three synthetic driver controls ensure every auxiliary/agent role is checked and bad storage stops before grading. These tests are not genuine-model qualification claims.

The prospective streaming coordinator is a separate source version. Its R2
per-role storage accepts exact-byte private offload as storage evidence only,
then reads each array back for independent numerical verification. Source and
job identity remain pinned. An unresolved SSH launch journal prevents duplicate
remote execution and prevents a restart from silently opening another epoch.
Training reports must match the exact signed job and approved objective.

Its final completion gate requires both before/after evaluations to cover all
16 tasks with zero errors, exact task hashes and seeds, verified finite rewards,
and the same authenticated dataset/fixed auxiliary descriptor. Partial results
retain training and successor-verification facts separately and cannot claim
full completion. The frozen first-service source remains unchanged; its status
fields must be assessed against actual artifacts rather than used as a substitute
for an independent full completion audit. Streaming deployment is still a
separate prospective gate.

### Reproducing offloaded duplicate submission caches

The first epoch's local `cumulative-0.zip` and `cumulative-1.zip` were duplicate
caches, reclaimed during operator disk pressure after exact private R2 readback.
Their signed `cumulative-{0,1}.zip-signed-cache-storage.json` inventories preserve
the immutable object key, complete size and SHA256. The final selection, signed
freeze receipts, full numerical/native audit reports and raw private R2 objects
remain available. Cache removal does not remove the submission evidence.

Use the operator-only helper with the trusted authority already pinned in its
CLI; bucket credentials remain local. Select a new private output location with
at least the object size plus 1 GiB free:

```bash
.venv/bin/python -m ops.rehydrate_native_tau2_cache \
  --inventory state/native-tau2-common-live/epoch-1790869388/cumulative-1.zip-signed-cache-storage.json \
  --bucket-config state/r2-direct.json \
  --output /operator/private/audit-copy.zip
```

The helper authenticates the inventory before reading its object, rejects keys
outside the exact private epoch/cache locations, bounds the read to 250 MB, and
requires exact length/hash before publishing a mode600 file. It refuses existing
outputs and removes a failed temporary download. Rehydration verifies storage
bytes; the separate signed model/native audits establish the computation and
original-environment result. The final cumulative ZIP is the deadline-selected
submission; the earlier cumulative cache is historical upload evidence.

Root exercised the helper against the actual signed final-cache inventory and R2
object: all 45,986,655 bytes matched SHA256, the output had mode600, and only that
new owned test copy was removed afterward. Evidence:
`state/root-audits/tau2-real-rehydration-check.json`. Seven focused authentication,
key-scope, bounds, failure-cleanup and publication controls pass.

### Explicit evaluation recovery and private diagnostics

The first post-training evaluation ended with eleven verified tasks and five
failures (indices 27–31). Its original report remains unchanged. An isolated,
separately signed recovery attempt covers only those failed indices, using the
same trained checkpoint, fixed auxiliary model, original tasks, seeds and
computation source. Recovery does not rerun the optimizer or relabel the original
failed report. Its own process/exit journals, admission artifacts, numerical and
native audits, and exact R2 storage inventories must be checked independently.
A successful generation alone is insufficient; a verified task whose storage
offload fails must not be reported as fully stored recovery evidence.

`ops.native_tau2_private_rejection_observer` can record bounded exception text,
traceback and disk headroom to a private mode600 signed sink. It delegates the
original diagnostic and preserves its return or exception, including when the
observation sink fails. The instrumentation has its own signed source pins;
model and native source/guards remain unchanged. Four controls cover forwarding,
bounded messages, private permissions and sink failure. Historical filtered
diagnostics cannot establish an exception message that was never recorded.
Keep these private traces out of public reports and dashboard data.

Evidence for the explicitly approved recovery source and received admissions is
under `state/native-tau2-common-recovery/retry-1790882934/`, including
`root-recovery-source-approval-check.json` and
`root-recovery-admission-storage-check.json`. These receipts state their partial
coverage and distinguish signed lineage/storage attestations from a fresh root
model or R2 byte recomputation.
