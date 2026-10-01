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
