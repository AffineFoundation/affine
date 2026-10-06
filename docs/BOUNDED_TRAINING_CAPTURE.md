# Bounded training-document capture

This prospective policy addresses training supply lost during bounded R2 capture.
It is implemented but not activated in the running f213 deployment. Historical
manifests retain four-reader FIFO behavior. Capturing a document does not verify
its inference or environment outcome; training remains explicitly unaudited.

An operator may include `learner_capture_policy` in the opening configuration:

```json
{
  "version": "bounded-parallel-token-capture-v1",
  "workers": 8,
  "max_document_bytes": 2000000,
  "max_inflight_bytes": 16000000,
  "completion_order": "first-completed"
}
```

The same validated policy is bound into the first signed epoch manifest and the
persisted gateway commitment binding. It requires explicit unaudited training,
v2 or v3 small commitments, and a bounded hourly phase policy. Workers must be
4, 8 or 16, with exactly workers × 2,000,000 raw document bytes. This is a raw
payload budget, not a bound on decoded Python objects or HTTP buffers. Each
read permits one oversize detection byte. HTTP pool capacity matches workers.

New capture processes completed reads promptly instead of waiting for an earlier
slow request, then refills the bounded worker pool until the existing cutoff.
Exact canonical bytes, declared size, full SHA and server upload time remain
required before immutable publication. Journal updates stay serial and follow
successful publication. Resume reuses captured original documents; infrastructure
deferral is not fraud. Late or malformed documents remain excluded.

Per-run receipts record attempts, successful publications, transient and structural
failures, maximum in-flight work, cutoff and deferred slots. Completion of CPU
controls does not establish live throughput: qualify and measure a prospective
source before activation, preserving optimizer lineage and old source admissions.
No deadline extension, old deferral rewrite, sampling contract or miner cap change
is part of this policy. Source-bound optimizer cache compatibility must be reviewed
when deploying a new scientific source; a cold restore must not be hidden as a hit.

## Prospective durable whole-stage policy

Version 2 also reduces full-history gateway rewrites during commitment admission
and token publication. It remains default off and is not active in f213:

```json
{
  "version": "bounded-parallel-token-capture-v2",
  "workers": 8,
  "max_document_bytes": 2000000,
  "max_inflight_bytes": 16000000,
  "completion_order": "first-completed",
  "journal_version": "fsynced-per-epoch-capture-v1",
  "state_checkpoint_documents": 16
}
```

The checkpoint interval must be an integer from 1 through 16. This policy has two
separate, private per-epoch journals beside `gateway.json`. The commitment journal
binds the original epoch, activated miner set, complete discovery digest, window,
source, checkpoint and quotas. Discovery is checkpointed and fsynced before its
journal is created. Each admitted row retains the original authenticated envelope,
SHA, size, ETag and server upload time. Replay rechecks its signature and bindings;
it does not read a mutable replacement commitment. Structural rejection decisions
are recorded separately. Unresolved infrastructure still prevents complete
commitment admission. Commitment reads retain their four-reader behavior.

The token journal binds the complete admitted commitment inventory. A validated
byte-and-path intent is fsynced before immutable publication; its exact commit is
fsynced after successful publication. Full-state checkpoints occur at most every
16 admitted decisions or successful documents, with a final file and directory
fsync before capture returns. Journal writes are synchronous. Successful commits
survive a crash before the next full-state checkpoint. An unresolved publication
intent is reconciled by reading and fully hashing only its original frozen object,
including after cutoff. Missing objects remain uncaptured; malformed frozen bytes
halt recovery. This does not permit a late staging upload or extend the window.

Journals require canonical owned paths, private ordinary files, single links and
exclusive ownership. Canonical bounded records use sequence and hash-chain checks;
the original header and context are validated before a torn final append can be
discarded. Append or fsync failure aborts that journal instance. Local hash chains
detect corruption; they are not independent remote authority signatures. Durability
depends on the filesystem honoring file and directory fsync.

Per-run diagnostics also distinguish replayed documents, reconciled publications
and full-state checkpoint counts. Existing version 1 and absent-policy behavior
remain unchanged. The existing postfreeze selector keeps all cheap-eligible inputs
in the audit and reward population while selecting at most 256 training documents
under one persisted draw. Report captured, eligible, selected and unselected counts
separately; increased capture does not mean every document was trained. The final
immutable commitment publication stage remains serial and must also be measured.

Real process-crash controls cover both phases and cutoff recovery. Owned temporary
benchmarks identify full-state serialization as a material bottleneck, but do not
establish production R2 throughput. A reviewed coordinator runtime and prospective
signed policy are required before activation. Historical receipts, eligibility decisions,
optimizer lineage and source admissions remain intact.

A separately signed `operator_overlay` in the durable learner policy can deploy
these coordinator changes while retaining the exact remote scientific source.
It binds the complete original inventory, every permitted CPU override, the
distinct overlay directory and the same capture policy in the configuration.
Unlisted files, changed scientific modules and ambiguous import paths are refused.
The runner clears previous coordinator imports and loads the pinned overlay.
Without this explicit policy it retains the original runtime behavior.

The overlay does not change the model, numerical sampling contract or remote
miner/verifier/trainer source. Keeping that scientific identity preserves the
existing optimizer-cache binding. Activate only after the current publication
continuation is terminal and its model and optimizer authority are committed;
then check the first new signed opening and measure actual capture throughput.
