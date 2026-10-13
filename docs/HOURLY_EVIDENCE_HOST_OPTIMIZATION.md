# Hourly evidence collection: bounded host optimization

The validator continues to use the original sole hourly writer, six-hour contribution half-life, current registered UID mapping, penalties, last-best fallback and transaction cursor. This host optimization changes neither the mining contract nor scientific verification. It delegates weight submission to that existing writer and does not create a second writer.

A read-only replay of the 2026-10-13 18:00 UTC evidence cutoff took 375.50 seconds for the historical reader and 128.67 seconds for the MiMo reader, exceeding the existing 480-second fresh-evidence budget. The original writer correctly retained its last valid assessment and still submitted weights. The problem was stale evidence, not absent chain submission.

The entry `ops/hourly_population_index_entry.py` authenticates a separate operator-signed host composition, checks exact parent/module pins and delegates the unchanged writer. It applies two optimizations through cloned function namespaces:

- `ops/population_index.py` replaces the historical quadratic eligible-pair join with an index over exactly the same four fields. It preserves record order, match multiplicity, duplicate rejection and the original malformed-value comparison path. All signature and complete population reconstruction checks remain.
- `ops/mimo_population_cache.py` reuses successful population authentication across the two scans within one MiMo evidence read. Its key includes the entire canonical signed envelope, authority and manifest guard. New or changed content follows all original checks. Errors are never cached, returned values are copies, and the cache is discarded after the read.

Neither optimization modifies frozen source files or globally replaces imported functions. No persistent admission cache, file-stat trust, partial fresh assessment or increased timeout is introduced. The original 480-second evidence and 720-second outer process bounds remain.

Validation includes exact equality for 166 genuine signed historical populations, 13 eligibility/malformed cases, and rejection parity for altered signatures, checkpoint and batch content. Five genuine MiMo populations matched exactly before/after caching; altered signature and payload controls were rejected. Unit tests exercise duplicate multiplicity, malformed values, authority/guard binding, mutation, defensive copies, uncached errors and cold restart. Independent full read-only shadows compare all cohort snapshots, not just normalized weights. Ongoing worker completions can change queue hashes and post-cutoff exclusion diagnostics between reads; those differences must be explained separately from scientific parity.

The final read-only comparison at the same cutoff reduced evidence processing from 506.02 to 436.69 seconds (449.31 seconds including diagnostic setup/output). All 170 scientific snapshots matched exactly, including penalties and cohort details. Live queue changes accounted for the remaining metadata and post-cutoff diagnostic differences. This leaves about 43 seconds within the unchanged evidence budget; production cache writes and future evidence growth still require monitoring.

An additional row-validation memo was investigated but excluded because its typed-key overhead did not improve runtime. The final scope remains the two optimizations above.

Installation requires a reviewed, signed host composition and a successful `--check`. Only the existing hourly service's entry is changed; keep its previous configuration for rollback and preserve its transaction cursor. Confirm fresh evidence and the normal hourly transaction after activation. Source publication alone does not activate the host patch.
