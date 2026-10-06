# Prospective trusted CPU service lifetime authorization

This is a proposal, not a deployment or an API/job authority change.

The actual API173 and seven-source auditor units both use `Restart=no`, with a
10-second systemd start limit and burst five. API173 still runs with its original
execution scope and process. That startup scope expires independently of the
running process. The V6 auditor has a two-hour startup scope, and its separate
read-only API172→173 validation chain has a one-day window. An unchanged service
can continue running after its startup window; a later restart currently needs
a fresh authorization. Neither unit currently provides automatic crash recovery.

`ops/running_api_validation.py` is a default-off prototype for automatic renewal
of **validation-only** records under an explicit signed policy. It preserves the
original execution scope, proves the original process started within its scope,
checks the same live PID/start ticks/invocation/argv, all CPU operator files,
config, signed source registry, source runtime files and queue inode. It emits
immutable signed versioned records and an atomic signed latest locator. Its
versions are incompatible with API execution launchers. It never starts or
restarts an API, changes a queue, or signs a miner proof. A future metadata
consumer must explicitly support this policy; the live V6 wrapper does not.

Validation renewal alone does not solve service restart. A separate, explicit
ROOT-signed `trusted-cpu-service-lifetime-authorization-v1` should authorize a
bounded supervisor to check and restart a named CPU service after its original
instance genuinely exits. Bind the host/UID, unit and persistent unit bytes,
launcher and full operator inventories, config and scientific source registry,
queue path/inode, local authority identity, bucket authentication configuration
and allowed service actions. Prohibit configuration or source-admission changes,
GPU execution, queue rewrites, report reissuance and altered scientific claims.
Keep it distinct from epoch manifests, worker leases, miner proof grants and
historical startup scopes. A separately signed policy generation can revoke or
replace service authorization; historical scopes remain immutable.

The supervisor should verify the current instance before any signal, gracefully
stop only the authorized unit, wait for process death, and restart only the
same pinned service. Use bounded exponential backoff and a circuit breaker for
repeated failures. New process/invocation attestation belongs to a new signed
service-instance record, never a rewrite of an old PID record. API readiness is
the rejected unauthenticated request with the new process still alive. Auditor
readiness is authentic new queued-job/persisted-plan progress; completed-tick
health remains a separate metric. Preserve durable unit files and exact rollback
commands before attempting a switch. No SIGSTOP of a SQLite writer.

Before deployment, qualify wrong source/config/registry/inode/authority, expired
or revoked policy, unrelated live process, crash-before-bind, repeated-crash
circuit breaking, graceful database-lock release, preserved queue/lease/job
provenance, and failure of the restart supervisor itself. The existing API173
request/report validation and all scientific worker backends remain unchanged.
