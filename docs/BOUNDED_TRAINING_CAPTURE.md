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
