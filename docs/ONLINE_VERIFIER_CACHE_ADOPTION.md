# Online adoption of historical verifier caches

Normal verifier completion already retires downloaded submissions after the
coordinator acknowledges the report and evicts obsolete receipted checkpoints.
Historical files created before those receipts existed need one explicit
ownership bootstrap. They must not be deleted by an unbounded directory scan.

`ops/adopt_online_verifier_cache_catalog.py` provides that bootstrap without
stopping healthy leased workers. It accepts only a ROOT-signed
`owned-verifier-cache-catalog-online-wrapper-v3` payload. The older online v2
files are unsigned preparation templates, not deletion authorization, and the
quiescent v1 helper still requires stopped readers.

First stage reviewed helper code in a separate CPU-only namespace, preserving
the running worker and its scientific source. Run preparation against a named
v2 template with `--prepare-with-operator-overlay` pointing to the already
running immutable wrapper namespace. Preparation only returns observations; it
does not create ownership receipts or remove files. ROOT then checks the exact
checkpoint inventories and durable R2 records and signs the fresh payload.
Apply the signed file with `--catalog` and `--authority` within fifteen minutes.
Never publish capabilities or private catalog contents in logs or Git.

The payload binds the worker PID and process start time, wrapper/worker/lifecycle
file hashes, its source-registry hash, exact owned root and model inventories,
and fresh inode/stat snapshots. An unrelated historical reader referencing the
workspace prevents application. Every model adoption and deletion uses the
same nonblocking checkpoint leases inherited by live backend children. Busy,
changed, symlinked, hardlinked and missing files are skipped. Only explicitly
adopted models can be removed by this bootstrap: a newer checkpoint downloaded
after observation remains untouched. Kept checkpoints become managed and can
later retire through ordinary worker completion.

Optional `completed_downloads` rows include the original signed verification
job, ROOT-attested coordinator acceptance with exact job/report hashes, and
fresh snapshots for its named submission files. The helper verifies the actual
unchanged local report and job bindings and acquires the checkpoint lease
before retiring only those listed inputs. Unacknowledged jobs, active jobs,
changed files, reports and diagnostics remain. This operation never changes
claims, queue history, signed scientific sources, model weights or credentials.

ROOT must refresh observations if the wrapper restarts, its registry changes or
the catalog expires. There is no automatic self-signing or conversion from an
unsigned template into deletion authority.
