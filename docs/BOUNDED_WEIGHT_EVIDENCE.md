# Bounded evidence loading for hourly weights

The optional operator path in `ops/current_assessment_compact_queue.py` reduces
memory use while retaining the current miner assessment and hourly transaction
rules. It changes no mining quota, proof format, sampler or training admission
rule. Miners need no software update for this operator optimization.

Each assessment reads the actual SQLite job, report, signed report request and
remaining row fields under a consistent read transaction. Every string is freshly
hashed with its byte length. Stored report digests, cached timestamps and mutable
row counters are insufficient substitutes for those bytes. This still reads all
selected evidence bytes; it is not an incremental storage scan.

An authenticated cache hit must bind those fresh row fingerprints, the exact
selected committed records, the original authority and worker/source policies,
the cutoff activation phase, and the admitted cache signer and implementation.
A miss reads the full original row, verifies it still matches the captured
fingerprint, and runs the original signature/source/report admission. A removed,
rewritten or replaced row cannot silently reuse a different observation. Cached
admission does not change duplicate-credit, penalty or current-registration rules.

The cache stores small signed admission results. Its size, entry count, free-space
reserve, file ownership and permissions are checked. Cache corruption or an
unavailable optional cache falls back to original admission; deadline failures
still propagate to the existing last-best assessment path.

The compact SQLite path bounds row count, string cells, row strings and scan
runtime. It bounds selectors before transferring them to Python and extracts
only numeric completion values. Duplicate JSON keys use the last decoded
**top-level** `completed_at`, matching `json.loads`; nested or similarly named
keys do not affect the cutoff. Invalid timestamps remain invalid. New compact
queue provenance fields are explicitly versioned and never relabeled as the
old full-JSON queue digest.

Operators enable this only through an authenticated writer policy that pins the
new modules and exact `compact_queue_evidence` profile. The default remains the
original full-row loader. The original signing context and admission-cache
policy are preserved. An upgrade retains the same reward state, transaction
journal, submission cursor, lock and sole timer. A previous policy's last-best
or already-created same-hour assessment must be explicitly pinned by original
signature, byte hash and original writer-policy hash; it is not re-signed as a
new observation. Numerical or evidence outages retain the best valid assessment,
never burn or substitute owner-only weights.

CPU controls cover raw-byte tampering, source and worker changes, corrupt caches,
row races and removal, oversized cells, duplicate-key cutoff semantics, exact
reward calculation, and pending/old-hour transaction continuity. Deployment also
requires comparing the complete original and compact evidence calculations on
the same frozen queue, population registry and source/reference context. The
only normalized comparison fields are the intentionally different queue
provenance digest schemas. Compact memory measurements do not establish faster
end-to-end assessment by themselves.

A read-only qualification on a frozen 21.4 GB queue, with the same registry,
source/reference context and cutoff, produced byte-identical calculated
assessments and identical evidence apart from those declared provenance fields.
The original path took 314.66 seconds with 22.12 GiB peak process RSS; the compact
path took 262.66 seconds with 2.53 GiB. Both used 4,875 authenticated cache hits and
1,436 full-admission misses. These sequential measurements are not a controlled
cold-cache benchmark. The main demonstrated improvement is lower memory use;
the complete assessment still takes minutes in this partially cached fixture.
