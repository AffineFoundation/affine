# MiMo host capture scheduling

The optional operator launcher `ops/mimo_host_capture_launcher.py` avoids repeating
full admission for unchanged, already captured miner commitments. It composes
with the existing spawn-safe `mimo_host_path_launcher.py`. Deploying this wrapper
requires an explicit signed operator admission; publishing these files does not
activate it. Miner contracts, sampling verification, native grading, and role
worker source archives are unchanged.

## What is reused

The original `Coordinator.collect` authenticates submissions, resolves native
task prompts, checks proof metadata, publishes immutable snapshots and admits
cumulative claims into its ledger. Repeating this for unchanged commitments at
every poll duplicates substantial CPU and storage work.

The host cache stores only successful original collections from its own process.
It is scoped to the coordinator directory and exact full manifest digest. On
subsequent complete commitment listings, bounded read-only threads fetch the
small current commitment objects. Reuse requires:

- Exact commitment bytes hashing to the originally admitted commitment digest.
- Original upload-window and observation-time checks.
- Exact unchanged local bytes of the authenticated capture record.
- The original coordinator lock and an unsealed population.

New or changed commitments, modified capture records, and process restarts take
the original full path. A changed input evicts its cache marker before collection,
including when that collection rejects or fails. Restoring old bytes therefore
requires a fresh original admission. Transport errors remain infrastructure
errors. Only commitment GETs are parallel; native resolution, ledger writes,
signing and publication retain their original sorted serial order.

The final complete listing after the epoch deadline is still required. A scan
begun before the deadline can miss a later valid submission; this optimization
does not extend the deadline or finalize a partial listing.

A reused record retains its original signature. Its observation-local `new_groups`
count can remain nonzero, whereas a redundant collection would re-sign it as zero.
Eligibility uses total groups and their immutable commitments/documents; those
and the ledger remain unchanged. The existing behavior of retaining an earlier
valid capture after a malformed overwrite is unchanged, and malformed updates
are never cached as successful observations.

## Operator admission

The owner-only configuration contains:

- `version: affine-host-capture-cache-launcher-v1`.
- `base`: the original qualified import-path launcher's configuration, including
  the exact frozen entrypoint and lifecycle configuration paths and hashes.
- `workers`: an integer from 1 through 16 (initial setting 8).
- `host_files`: SHA256 pins for `mimo_host_capture_launcher.py`,
  `mimo_host_capture_cache.py`, and `mimo_host_path_launcher.py`.
- `original_methods`: SHA256 of `inspect.getsource` for `Lifecycle._capture`
  and `Coordinator.collect` before installing the wrapper.
- `admission`: an authority-signed envelope whose payload is:

```json
{
  "version": "affine-host-capture-cache-launcher-v1",
  "configuration_sha256": "SHA256 of canonical config excluding admission",
  "source_unchanged": true,
  "scope": "host-capture-scheduling-only"
}
```

The signing authority must match the original release policy. The wrapper first
runs the original launcher checks, including its full source inventory guard.
It then authenticates the operator admission and exact method/file hashes before
installing the disclosed host scheduling override. No scientific file is edited;
this does not mean host runtime scheduling is unchanged.

Review with:

```sh
python -B ops/mimo_host_capture_launcher.py --operator-config /private/operator.json --check
```

Check mode does not install the override or start services. After operator review,
change only the host lifecycle entrypoint, preserving its original state, signed
policy, configuration and API address. Remote roles and the hourly weight writer
continue independently. `capture-cache` log records report listed, reused,
full-collection and rejected counts. Restoring the prior host launcher is the
rollback; no database migration is needed.

## Validation and limits

The tests cover unchanged reuse, cumulative updates, restart, altered records,
late overwrite, malformed data, read outages, rejected-then-restored commitments,
complete listings and sealed populations, plus launcher admission and pin checks.
An isolated replay of actual epoch 190 two-group inputs used production GET/HEAD
only. Initial signed captures and ledgers matched; warm reuse differed only in
`new_groups` as described above. Repeated original processing took 1.446 seconds
versus 0.0152 seconds for reuse on local object transport. This measures avoided
local work, not end-to-end production speedup or R2 latency.

This change does not remove the separate synchronous audit-population construction
before trainer enqueue. Improving that path requires preserving durable audit
reconstruction/retry rather than launching an untracked background thread.
