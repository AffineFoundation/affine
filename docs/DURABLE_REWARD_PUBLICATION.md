# Prospective durable training before reward eligibility

Opt-in signed manifest field:
`reward_publication_policy = durable-training-before-reward-v1`.
Old manifests without this field retain their original semantics; no old E10
scores, writer cursor, source grants or paid history are edited or relabeled.
Final live source/qualification/ordered source authorization remain root gates.

The synchronous controller emits `epoch-reward-publication-ready.json` only
in its after phase, after the actual trainer returns, durable publication succeeds,
the latest optimizer journal matches the output and independent evaluation is
queued rather than awaited. It obtains and verifies actual ROOT-signed optimizer
and checkpoint descriptors from R2, binds the original signed training job/report,
validates persistent report receipts/updates and requires the actual root checkpoint
publication receipt with every declared file SHA. The public ROOT-signed readiness
is read back before saving the local attestation. A public-PUT/local-save interruption
recovers exactly that original signed attestation without new timestamps or overwrite.

Binding chain: exact original signed manifest/source/score SHA → input checkpoint,
parent pointer/counter and training binding SHA → original training job ID/SHA,
source inventory and runtime digests → real training steps and advanced optimizer
counter → ROOT-signed output optimizer descriptor and checkpoint descriptor →
actual immutable checkpoint publication receipt and metrics SHA. No GPU recomputation
is performed by this gate. Cryptographic scientific verification and selected worker
lineage remain the original audited/receipt pipeline, not replaced by this attestation.

The writer's completed-hour completeness check defers while the new-policy readiness
is absent. The direct exporter also rejects early eligibility. Tampered/missing
originals, job lineage, signed descriptors, advanced counters, publication receipts
or readiness signatures fail closed. Source-specific authorization still applies.
Historical completed readiness remains valid after later optimizer steps; only the
initial emit requires the current latest journal to equal that epoch's output.

An empty epoch requires explicit `closed_no_accepted_batches`, zero points and a
signed `closed_no_update` readiness: zero updates, unchanged checkpoint, no invented
optimizer publication. This permits honest zero-credit hour closure, but is never
evidence of convergence or a complete nonempty training-hour goal.

UTC window inclusion remains the original score `finalized_at`. The writer can
close that original window only after new-policy readiness, so chain submission
happens after durable training publication. It may be delayed past the wall boundary;
no completed hour or chain receipt is fabricated and no old cursor is reset.
To achieve <=3600-second full epochs, root still needs measured wall-hour pacing
and sufficient publication throughput; this gate provides ordering, not a timing
guarantee. More than one trainer/reader/writer must not be started on an observation
timeout. Late original jobs and immutable attestation recovery are retained.
