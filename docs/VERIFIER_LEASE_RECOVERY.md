# Verifier recovery after an expired completed lease

The verifier keeps its original backend running until that subprocess exits.
If the lease expired meanwhile, it preserves the original attempt diagnostics
and records `expired-completed-lease.json`. It does not submit the unacknowledged
report, acknowledge that attempt, or restart its physical execution. The worker
returns to normal coordinator polling; only the coordinator may assign further
work under its existing attempt and lease policy.

The same rule applies when report acknowledgment fails after the known lease
deadline. Exact pending report bytes and downloaded inputs remain available.
An expired lease is infrastructure evidence, not a miner fraud verdict.
The diagnostic contains the original job hash, attempt, terminal backend exit,
stage and lease deadline, without the bearer token.

Only the typed `ExpiredCompletedLease` condition is recoverable this way.
Signature, source, integrity and unclassified failures still stop the worker.
This change does not weaken sampler, TOPLOC or environment checks.

The main branch implementation has controls for expiration during backend
execution, expiration during report acknowledgment, continued polling and
unchanged fatal authority checks. Existing immutable backend archives are not
modified by committing this code. A live parent-worker deployment requires its
own reviewed package or operator-only overlay and an idle handover.
