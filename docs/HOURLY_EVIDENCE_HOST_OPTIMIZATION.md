# Hourly evidence collection: bounded host optimization

The validator continues to use the original sole hourly writer, six-hour contribution half-life, current registered UID mapping, penalties, last-best fallback and transaction cursor. This host optimization changes neither the mining contract nor scientific verification. It delegates weight submission to that existing writer and does not create a second writer.

A read-only replay of the 2026-10-13 18:00 UTC evidence cutoff took 375.50 seconds for the historical reader and 128.67 seconds for the MiMo reader, exceeding the existing 480-second fresh-evidence budget. The original writer correctly retained its last valid assessment and still submitted weights. The problem was stale evidence, not absent chain submission.

The entry `ops/hourly_population_index_entry.py` authenticates a separate operator-signed host composition, checks exact parent/module pins and delegates the unchanged writer. It applies the following bounded optimizations:

- `ops/population_index.py` replaces the historical quadratic eligible-pair join with an index over exactly the same four fields. It preserves record order, match multiplicity, duplicate rejection and the original malformed-value comparison path. All signature and complete population reconstruction checks remain.
- `ops/mimo_population_cache.py` reuses successful population authentication across the two scans within one MiMo evidence read. Its key includes the entire canonical signed envelope, authority and manifest guard. New or changed content follows all original checks. Errors are never cached, returned values are copies, and the cache is discarded after the read.
- `ops/no_eviction_cache.py` prevents the optional legacy cache from evicting entries during a scan that may still need them. At capacity, new records follow normal uncached validation and optional writes stop. Existing entry signatures, policy bindings, size limits and disk-space checks remain enforced. Only the invocation-owned cache instance is affected, and its original method is restored on exit or timeout.

These optimizations do not modify frozen source files or globally replace imported functions. No new persistent admission format, file-stat trust, partial fresh assessment or increased timeout is introduced. The original 480-second evidence and 720-second outer process bounds remain.

Validation includes exact equality for 166 genuine signed historical populations, 13 eligibility/malformed cases, and rejection parity for altered signatures, checkpoint and batch content. Five genuine MiMo populations matched exactly before/after caching; altered signature and payload controls were rejected. Unit tests exercise duplicate multiplicity, malformed values, authority/guard binding, mutation, defensive copies, uncached errors and cold restart. Independent full read-only shadows compare all cohort snapshots, not just normalized weights. Ongoing worker completions can change queue hashes and post-cutoff exclusion diagnostics between reads; those differences must be explained separately from scientific parity.

The final read-only comparison at the same cutoff reduced evidence processing from 506.02 to 436.69 seconds (449.31 seconds including diagnostic setup/output). All 170 scientific snapshots matched exactly, including penalties and cohort details. Live queue changes accounted for the remaining metadata and post-cutoff diagnostic differences. This leaves about 43 seconds within the unchanged evidence budget; production cache writes and future evidence growth still require monitoring.

An additional row-validation memo was investigated but excluded because its typed-key overhead did not improve runtime. The row memo is not part of the deployed or proposed path.

Installation requires a reviewed, signed host composition and a successful `--check`. Only the existing hourly service's entry is changed; keep its previous configuration for rollback and preserve its transaction cursor. Confirm fresh evidence and the normal hourly transaction after activation. Source publication alone does not activate the host patch.

## Live cache churn found at 19:00 UTC

The initial two optimizations did not eliminate the production timeout. The 19:00 writer still submitted weights for 250 UIDs using its last valid assessment. Its optional cache held 10,000 records, while the scan needed about 10,092. Chronological reads and oldest-entry eviction created a cascade of 1,024 new writes over approximately 102.5 seconds. The earlier read-only shadow did not exercise that publication behavior.

The version-2 host composition adds the source-pinned no-eviction adapter above. The signed cache remains limited to 10,000 entries and 256 MiB, with the same 8 GiB free-space floor. Increasing the signed cache policy would invalidate existing cache keys, so this repair preserves the warm entries. Portable tests include a reproduction of the eviction cascade, identical scientific admission results, signature tampering, malformed evidence, source pins, instance isolation, timeout cleanup, free-space and byte limits. The original storage implementation is published as `ops/current_assessment_cache_storage.py` so these tests are reproducible outside the operator host.

A cache miss still incurs full original validation. Further growth in new records or MiMo history can consume the remaining time margin; this repair is not an unlimited-throughput guarantee. Historical hourly assessments and already-submitted chain transactions must not be rewritten to make a repair appear retroactive.

The production-equivalent 19:00-cutoff qualification used the original authorized cache producer, with 10,000 hits, 92 misses, zero evictions and zero writes at capacity. Complete evidence processing took 444.00 seconds, with calculation and validation taking another 0.17 seconds. All 171 scientific snapshots, per-miner estimates and points matched the read-only reference exactly. Queue changes between reads affected only explicitly reviewed metadata and post-cutoff diagnostics. This tests the real write-capable path, not merely a read-only approximation. The next scheduled live hour must still confirm fresh assessment and submission after activation.


## Population derivation reuse

The v3 host composition also pins `population_derivations.py`. After admission
of the original source inventory, this helper reuses the manifest digest and
cohort derivation within one population reconstruction. It retains the original
signatures, release guard, commitment and native-grade checks, and canonical
reconstruction comparison. An isolated copy of the opening prevents caller
aliases; unexpected guard types use the original path. No authenticated result
is cached across invocations.

Six actual signed populations reconstructed identically in 50.38 seconds versus
74.75 seconds originally. This measures population reconstruction only, not the
whole hourly assessment. The signed host policy must opt into the new helper;
committing this code does not activate it. The same 480-second evidence budget,
720-second outer bound, admission-cache limits, scoring, current registrations,
transaction cursor and last-valid-assessment fallback remain in force.

The composed MiMo reader was also compared against its predecessor on the same
19:00 UTC cutoff and six original population documents: 110.22 seconds versus
84.24 seconds, with byte-identical whole evidence output. The unchanged legacy
reader was not rerun in that comparison. An actual natural hourly assessment
and chain submission remain the operational check; these timings do not promise
unbounded history growth within a fixed deadline.

## Reuse commitment derivations within one population

The v4 host composition adds `ops/commitment_derivations.py`, admitted by an
explicit `commitment_module` path and SHA256. It reuses the original sampling
context and scope once per miner, and the original environment entry once per
environment, while reconstructing one closed population. The existing wrapper
first isolates the signed opening with a private copy. Cached values never
persist to another population or reader invocation.

Original miner signatures, registration checks, commitment digests, every claim's
slot/index/harness checks, byte limits, native grades and canonical population
comparison remain in force. Only the exact pinned release guard and source are
eligible; malformed environment IDs retain original behavior. No miner upgrade,
sampling-contract change, scoring change or timeout increase is involved.

In a private comparison using eight original closed populations and one captured
read-only queue snapshot, the complete MiMo evidence reader took 121.3 seconds
before and 72.6 seconds after this change. Its entire evidence output was identical,
including snapshots, hashes, refusals and deferrals. Population reconstruction
alone fell from 75.6 to 27.3 seconds. These are measured diagnostic results, not a
promise for future history sizes: all historical populations are still read, and
new reports add work. A subsequent complete historical-plus-MiMo dry run
reconstructed 174 snapshots and calculated and validated a fresh assessment in
447.9 seconds under the original 480-second evidence budget, leaving 32.1 seconds.
It made no chain transaction or writer-state change. That margin is limited;
continued history growth can exhaust it, so natural hourly freshness still needs
monitoring. Activation requires a separately signed host composition. Public
source availability does not itself activate the wrapper.
