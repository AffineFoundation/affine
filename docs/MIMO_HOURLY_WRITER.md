# MiMo evidence and the existing hourly writer

`ops/mimo_hourly_writer.py` is an explicit composition entry for the existing
hourly writer. It calls the original `writer_entry.prepare` and
`current_assessment_writer.run_once(evidence_loader=...)`. It retains the signed
parent writer policy, six-hour EMA, current-registration mapping, transaction
cursor, global lock, prior assessments and last-best fallback. It does not
implement a second writer or a burn path.

The operator signs a `mimo-existing-hourly-writer-evidence-composition-v1`
payload with these exact fields:

* `execute_enabled`: initially `false`; execution also requires `--execute`.
* `entry`, `legacy_entry`, `legacy_composition`, `evidence_module`,
  `reader_policy`: exact `{path, sha256}` references.
* `new_source_root`: absolute path to the immutable reviewed MiMo release.
* `new_source_files`: SHA-256 map for every Python file under its `subnet/`.
* `allow_public`: explicit local public policy opt-in.
* `maximum_seconds`: `720`, retaining the existing process deadline.

`legacy_composition` is the unchanged signed composition used by the current
sole writer. Its parent policy, cutover and anchor are passed through unchanged.
`evidence_module` must identify the already-loaded original
`ops.current_assessment_evidence`; its `load_evidence`, `snapshot`,
`current_estimates` and `observations` are reused directly. The new release is
loaded under a separate package alias, so it cannot replace the historical
`subnet` package or old sampling functions.

The signed reader policy follows `mimo-current-assessment-evidence-bridge-v1`:
it binds the durable scheduler directory, read-only queue, qualified release,
approved verifier execution closures, audit policy, admitted verifier IDs and
explicit conservative overlap policy. The first cutover reuses verifier keys
already admitted by the original writer policy. Additional identities require a
separate reviewed writer-policy and fallback-history handoff; this entry refuses
them instead of changing that history implicitly.

`--check` authenticates the composition without reading wallet credentials,
writing assessments, or submitting chain transactions. Only after private
queue evidence and a read-only assessment comparison pass should the existing
sole service's entry be switched to this reviewed entry under a signed enabled
policy. Keep the existing timer, lock and state paths. Never start a second
writer. A reader failure propagates into the original authenticated last-best
fallback. A same-hour cached assessment can defer new evidence until the next
hour; it is not overwritten.

The conservative transition takes the stricter legacy/MiMo confidence and
penalty factors during overlap. This is an explicit temporary transition policy,
not a mathematically pooled posterior. Its retirement must be reviewed so a
frozen historical factor cannot constrain new cohorts indefinitely.


## Temporary diagnostic deferrals

`ops/mimo_quarantine_writer.py` can wrap the existing sole writer under an
explicit ROOT-signed composition. It authenticates the original MiMo evidence
first, then defers only individually named reports bound to their original job,
report, signed request, worker, miner, epoch, checkpoint and diagnostic digest.
The original verifier verdict and signed artifacts remain unchanged.

This is an evidence dispute mechanism, not a new sampling acceptance rule.
Deferred reports neither count as verified successes nor impose invalid-sample
penalties or numerical-resolution penalties. All other observations, eligible
batch counts and the existing confidence estimator continue unchanged. The
public contract's sampling tolerances are not widened.

Each disposition has a prospective hourly cutoff, explicit reason and bounded
expiry. The writer records the disposition in its deferred evidence. Expiration
requires resolution or a newly reviewed disposition; it does not silently turn
an unresolved diagnostic into fraud. If the evidence reader cannot proceed, the
original hourly writer retains its authenticated last-best assessment. Its
transaction cursor, six-hour EMA, current UID mapping and no-burn behavior are
unchanged. Never start a second writer to apply a disposition.
