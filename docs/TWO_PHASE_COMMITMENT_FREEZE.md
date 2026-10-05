# Complete metadata capture before bounded artifact copying

This prospective source change addresses a measured E11 ordering problem: large
copies for early sorted identities exhausted the 60-second freeze budget before
the owned miner's already-uploaded small commitment was journaled. Historical E11
receipts are unchanged.

Freeze now closes writes and fully paginates the exact epoch commitment prefix.
The list is advisory discovery only: foreign keys are ignored, and every listed
activated identity still requires an atomic bounded GET, its original ETag and
LastModified, the miner signature, approved checkpoint/source, ordered slots and
signed batch cap. The signed envelope must use exact canonical JSON bytes
(`sort_keys=True`, compact separators, no NaN); pretty-printed or differently
encoded copies of the same signed payload are structural rejections before
journaling. This binds the raw committed SHA to the parsed-envelope SHA used by
the downstream queue, preventing a noncanonical envelope from blocking every
audit. Official clients already publish this encoding. At most four small GETs
run concurrently. A failed or malformed
list, interrupted pagination, timed-out GET, or unvisited discovered identity is
an infrastructure-incomplete exception. It cannot finalize an empty or partial
receipt roster. Malformed miner documents remain policy rejections.

Every authenticated tiny document is durably journaled before any heavy copy.
After complete capture, a random copy ordering is generated once and persisted;
recovery reuses it. Original artifact HEAD ETags/size/time are journaled before
dispatch. Up to four independent miners can copy concurrently using the saved
conditional ETags. Root journal updates remain serialized; active calls are
observed to completion even when the cutoff stops new copy launches. A transient
copy error preserves actual completed copies and the original pending plans;
retry never rereads or replaces the committed tiny document. A stale conditional
ETag is infrastructure evidence, not a fabricated fraud finding.

The signed cutoff still limits new heavy copies. Authenticated but incompletely
copied submissions can therefore be infrastructure-deferred, without credit or
fraud penalties. This is not a claim that all payloads fit in 60 seconds, nor a
protocol change to selected-only freezing. Full public receipt publication and
selected verifier SHA/readback rules remain unchanged. Only a newly reviewed and
sealed source can activate this behavior.

If complete metadata capture is still impossible after the explicitly signed
hourly freeze cutoff, the remote controller closes that epoch as
`infrastructure_skipped_metadata_incomplete`. It publishes a signed
`capture-status.json` with `complete:false`, known captured commitment hashes,
unresolved identities and infrastructure evidence; it never fabricates an empty
`receipts.json`, audit challenge, score or accepted-data claim. A separate signed
infrastructure-skip history records the zero-update closure. Only the epoch
counter advances: checkpoint, checkpoint path, optimizer parent and public
training counters are unchanged. No inference, training, evaluation or reward
dispatch is required for this closure. A failed status publication retries the
same original signed evidence before advancing. Historical manifests without
the explicit hourly cutoff keep their existing retry behavior.

Controls cover a 246-identity roster with only one present object, exact paginated
namespace filtering, failed pagination/GET fail-closed behavior, four real
overlapping copy threads with serial journals, complete metadata capture before
an expensive first copy, stale ETag recovery, and persisted random order.

An isolated actual R2 control uses 246 generated test identities, four signed
opaque-payload commitments (about 2 MiB total), a malformed registered document
and foreign keys. It records one LIST, five tiny GETs, bounded overlapping actual
conditional copies (peak three on the final source, with one injected request
failing before transport), all valid tiny journals before any copy, original-ETag retry
after an explicitly operator-injected pre-copy transient, and complete frozen
byte/SHA plus public receipt readback. The private qualification evidence lives
under `state/root-audits/two-phase-freeze-actual-R2-qualification-20261005-v1`.
This is storage transport evidence only: no model, TOPLOC, GPU, chain transaction,
production manifest mutation, or inference-validity claim is involved.
