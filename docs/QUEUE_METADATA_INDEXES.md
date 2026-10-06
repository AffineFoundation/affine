The coordinator now creates four metadata indexes when it initializes a new
or existing queue. Expiration, failed-lease selection, claim selection and
nonce expiration use status/role/time metadata instead of scanning retained
signed envelopes and reports. The existing claim SQL, rowid ordering, schemas,
worker authentication, report admission and lease semantics are unchanged.
Existing job, report, request and event rows remain intact. A conflicting index
with an approved name is refused rather than silently replaced.

Initialization runs each schema/index statement inside the same transaction;
SQLite executescript would otherwise implicitly commit the initial BEGIN.
Queue acquisition retries only locked/busy BEGIN IMMEDIATE failures, using a
120-second maximum and short waits. It checks the queue device/inode before
and after acquisition and before commit. A replacement or missing queue is
refused. Bodies and commits are never replayed: their failures roll back and
propagate normally. A failed lock acquisition is not an idle-lease assertion.

This change is prospective code. Deployment to a pinned operator tree still
requires its own admission. The October 6 operator index-only migration and
scoped auditor retry were independently reviewed and applied without changing
scientific job sources or rewriting historical reports. Query-plan and
contention controls cover the actual claim predicate, preserved selection and
rows, legacy initialization, conflicting indexes, lock release/timeouts,
transaction-body/commit errors, inode replacement and concurrent claims.
