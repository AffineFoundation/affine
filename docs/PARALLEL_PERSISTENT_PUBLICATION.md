# Prospective parallel publication and independent CPU readback

No running controller, epoch, source, optimizer, checkpoint or verifier roster is
changed by this code. An absent signed policy retains serial checkpoint then
full optimizer-state readback. The future opening may opt in explicitly:

```json
{"persistent_publication_policy":{"version":"parallel-persistent-publication-v1","state_readback":"qualified-remote-full","checkpoint_readback_workers":4}}
```

The immutable signed training execution scope preserves this policy. Worker
upload jobs with the policy admit at most four BF16 PUT streams; the operator
admits at most four complete checkpoint GET/hash streams. Every checkpoint file
must pass its exact original SHA and complete advertised size before checkpoint
descriptor publication. A malformed response, incomplete file or encoded object
cannot produce a checkpoint signature.

The persistent controller starts checkpoint publication and optimizer readback
concurrently, without issuing a second training job or re-verifying training
rollouts. Optimizer authority state is signed only after both paths succeed.
Neither a missing reader nor an infrastructure failure silently switches hosts
or resets optimizer lineage. An incomplete attempt leaves the original journal
unchanged. A completed original training report remains the recovery anchor.

`state_readback: "local-full"` is an explicit alternative with the same overlap
and complete SHA checks through the operator. It does not remove the operator's
91 GB transfer. `qualified-remote-full` uses the separate CPU reader and scoped
key; only the small original request, receipt and process evidence reach Arbos.

## Qualified reader configuration

`remote.independent_state_reader` is deliberately absent from current configs.
Its exact schema consists of endpoint, reader_host, trainer_host,
trainer_known_hosts, reader_identity, module_hashes, qualification and
max_wall_seconds. The endpoint fields are host, port, user, known_hosts, python,
workspace (the existing evaluator backend), and namespace (the pre-staged reader
code). Physical records contain provider_UUID, ssh_host_key_sha256,
evidence_sha256; actual trainer and reader UUIDs must differ. The scoped reader
seed remains `namespace/reader.seed` on the reader, mode0600, never on Arbos.

Root must independently review the actual qualification evidence before signing
the admission payload, which is exactly:

```json
{"version":"qualified-independent-state-reader-v1","reader_identity":"<64hex>","reader_host_record_sha256":"<64hex>","module_hashes":{"remote_optimizer_readback.py":"<64hex>","helper.py":"<64hex>","supervisor.py":"<64hex>"},"all_23_objects_full_hash":true,"qualification_evidence_sha256":"<64hex>","qualified_at":0}
```

The qualification timestamp must be real. Root's admission envelope authenticates
its reviewed full-object qualification, not a self-certification by the reader.
Reader receipts remain trusted operator-host attestations, not cryptographic
execution proofs against a malicious host.

The dispatcher verifies original signed job/manifest, scientifically validates
the full descriptor (including all tensor metadata), fetches the exact staged
descriptor and projects only name/size/SHA into the reader inventory. It verifies
exact current UUID-scoped host-trust bytes and remote modules/key; both pre-sign
and immediately pre-spawn admission require no current evaluator/GPU job. Signed
GET-only capabilities expire within3500 seconds. Root transmits only signed
request/launch envelopes, lowers CPU priority to19, clears CUDA_VISIBLE_DEVICES,
and uses the existing four-stream1MiB reader. No worker enrollment occurs.

The original root/remote dispatch namespaces are exclusive. Root preserves the
original PID/ticks, real child wait and complete receipt. A failed or ambiguous
dispatch never silently retries. This initial integration fails closed and
requires explicit original-process recovery if an attempt already exists; it
does not yet automate that recovery. The authority adapter durably persists and
reads back authenticated evidence before publishing optimizer authority last.

## Actual measurement and remaining qualification

On2026-10-05 a distinct retained evaluator CPU read all23 original E10 optimizer
objects (91,387,491,264 bytes) in278.0755 seconds. The original supervisor/child
wait ended exit0 after278.2325 seconds; every original size/SHA, job, source,
descriptor, signed request and scoped identity matched. CPU priority was19;
one live observation showed34,740kB RSS. No GPU/model cache or authority commit
was used. The durable control evidence key is
`private/root-transport-qualification/independent-reader-original-E10-20261005-v5/evidence.json`,
SHA256 `5e9477430bb04286d14c151851cf0bcce420c47e42f4e24e48659daf4a32f632`.
Root qualification admission and deployment remain separate steps.

Original E10 runner→report was5474.477 seconds. Actual final BF16 file mtimes
bound all pre-state-export work to768.708 seconds; subsequent state export,
trainer PUT/full GET verification and descriptor staging consumed4705.769
seconds (85.96%). These are actual boundary observations, not instrumented pure
forward/CPU-optimizer timings. E10 used genesis and restored no parent state.
The interrupted separate four-stream export/restore control has no full success
and supplies no completed production export timing.

Future worker diagnostics separately record parent restore, reference forward,
gradient/clip, CPU optimizer by step, post-update forward, BF16 save, and state
export/trainer readback durations. CUDA phase timers synchronize only to measure
boundaries; arithmetic, clipping, FP32 state, update counts and reward rules are
unchanged. This changed source still requires normal role/GPU qualification.

One signed step per epoch is already supported by the optimizer. E10's58/57/57
groups partitioned172 tasks, so one step with172 accepted tasks removes two CPU
optimizer passes while retaining all172 task forward/backward computations.
A smaller prospective audited cohort is also needed to reduce those computations.
Use the actual committed step3 parent and advance to4, never reset genesis or
rewrite E10. No one-step throughput, convergence or <=3600-second epoch is
established by the CPU transport measurement.

Original dispatch recovery now reuses the retained signed request and launch,
exact namespace and original supervisor PID/start ticks. It does not sign a
replacement, copy a replacement envelope or start another process. A transient
SSH observation failure leaves these files intact and polls the original handle;
a controller retry reconstructs the binding from the same job/report/manifest and
observes that original process. Full receipt, copied-envelope hashes, actual
child-wait PID/ticks and host trust bytes still have to pass. Missing original
handles, partial pre-launch preparation, changed records and expired capabilities
remain explicit operator recovery gates. This does not extend an expired signed
request or turn a lost observation into a successful result.

For a reader hosted on the evaluator, the initial fresh idle gate alone is not
mutual exclusion with the independent checkpoint evaluator. Activation must
reserve that evaluator for the CPU reader until the original readback terminal,
or select a distinct dedicated CPU reader host and qualify its scoped signer and
actual full-state readback. A local flock alone is insufficient after a controller
crash while its remote child remains live. This implementation does not weaken the
idle gate or claim that scheduler coordination exists. Resuming an already
admitted CPU-only reader does not start a GPU workload and does not require a new
idle probe; any later evaluation overlap must be labeled in performance evidence.
