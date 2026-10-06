# Trusted CPU service recovery, default off

`ops/trusted_cpu_service_lifetime.py` supports a distinct ROOT-signed lifetime
authorization for the existing queue API and continuous auditor. It does not
change an epoch manifest, a miner/verifier proof contract, a historical startup
scope, a job, or a lease. No deployment has been made by this change.

The authorization pins the host/UID, private local authority, full API173 and
auditor2102 operator trees, configs, signed source registry and complete runtime
maps, R2 credentials file, original queue inode, persistent unit files and exact
entrypoint argv. Historical scopes and their actual initial instances remain
immutable provenance. A signed authorization-status record can revoke or replace
the lifetime grant. The default `execute_allowed=false` permits only validation.
An explicitly approved lifetime grant has no routine startup-window expiry; it
is separate from the short windows governing the preserved original scopes.

The new CPU supervisor adopts the exact original process without interrupting
it. Only after authoritative process absence does it start the same reviewed
unit. API readiness requires the real rejected empty unauthenticated request.
Auditor readiness requires genuine signed queued-job/persisted-plan progress or
a genuinely completed new tick; these are labeled separately. Every new instance
and restart event gets a signed, immutable local record. A single-owner lock and
the signed history preserve the restart circuit across supervisor crashes.

Restart attempts, including processes that bind successfully and then crash,
are bounded to three per 30 minutes in the prepared profile, with bounded
backoff. Failed startup cleans up only the exact new owned CPU instance. It
preserves original and unrelated processes. Config/source/credential/inode
changes, revocation, or uncertain identity stop recovery rather than changing
claims. Queued and leased verifier jobs continue to use the unchanged API guard
and scientific backends. API/auditor source maps can change only through a new
explicitly reviewed authorization and source admission.

Prepare an immutable operator package with
`ops/prepare_trusted_cpu_service_lifetime.py --output ABSOLUTE_FRESH_DIRECTORY
--authority PUBLIC_AUTHORITY`. Preparation only reads the existing services and
writes new review files. Its two prepared drop-ins select the new lifetime
entrypoint on a future restart; they do not restart the current services.

After ROOT reviews the full package and controls, ROOT can run
`ops/approve_trusted_cpu_service_lifetime.py --package DIRECTORY --authority
PUBLIC_AUTHORITY --sign`. It writes the new signed lifetime authorization and
status O_EXCL. ROOT activation uses `ops/activate_trusted_cpu_service_lifetime.py
--package DIRECTORY --authority PUBLIC_AUTHORITY --activate`; it installs only
the reviewed metadata files and starts the two new CPU supervisors. It never
stops or restarts the original API/auditor during this handoff. Activation is
confirmed by the supervisors' actual identities and signed instance records,
not merely systemd's active state. Preserve all previous attempt evidence.

CPU controls cover exact original adoption, process/PID reuse, changed pins,
revocation, unbound startup, timeout reconciliation, bounded failure and
short-lived-success circuits, source/queue/signature binding, real HTTP
rejection, SQLite busy behavior and actual full-tree constructor assembly.
The science trees and current services remain unchanged during qualification.
