# Durable pinned CPU audit services

`ops.durable_audit_services` runs the independent API and continuous auditor from a stable ROOT BASE64-signed execution policy. The policy has no clock expiry, process ID, invocation ID or boot ID. Restarting after a crash or reboot therefore requires no renewed signature. This does not extend any scientific job, lease, capability or admission deadline.

Every startup verifies the runner bytes; local UID and machine-id digest; exact config bytes; original queue device/inode, schema and WAL/FULL mode; unchanged operator files; ROOT-signed historical and current source admissions; private authority-seed identity; and all source/overlay hashes. API policy preserves eight historical registry rows and adds exactly one source. Auditor policy preserves the old source maps and execution-evidence cutoffs, and imports the existing V13 BEGIN/status retry functions and three frozen CPU overlays. No transaction-body replay or score reinterpretation is introduced.

A private, owned, single-link singleton file lock stays held for the entire service lifetime. The policy lists predecessor units which must be inactive before startup. API and auditor use different locks. The auditor also requires an active durable API unit whose actual process names the exact signed, pinned API policy. This dependency follows restarts without binding a changing PID or expiring handoff scope.

Policy payload fields are exact: `version`, `kind`, `execute_allowed`, `authority`, `identity`, `config`, `queue`, `singleton_lock`, `excluded_units`, `runner_file_sha256`, `operator`, `admission`, `historical_admission`, `source`, `source_trees`, `authority_seed`, `API_dependency`. Version is `durable-pinned-audit-service-v1`; kind is `API` or `auditor`. Config binds path/file SHA. Queue binds path/device-inode/schema SHA (`queue_schema` canonical digest). Each admission binds path/file SHA/payload SHA of an existing BASE64 ROOT-signed document. Identity binds UID and SHA256 of `/etc/machine-id`. Operator binds root/file inventory, nullable retry helper and overlay. Seed binds the existing path/file SHA without exposing its bytes. API dependency is null for API; for auditor it binds unit/policy path/policy file SHA. Source trees are all nine exact approved roots for API and empty for auditor.

Prepare API policy first, ROOT-sign it, then bind its actual signed file SHA into the auditor policy. Preserve old configs, signed policies, admission documents and queue. Gracefully stop only the old CPU API/auditor owners, never a GPU backend or worker. Install enabled systemd user services with `Restart=on-failure` and the unchanged API/auditor config, then start API before auditor. The execution command is:

```
python -B -m ops.durable_audit_services --policy /absolute/policy.ROOT-SIGNED.json
```

`--check` performs startup validation and CPU imports under the same singleton lock without starting the service. It requires predecessors inactive. Run each service from a separately frozen/pinned operator runtime, not a mutable checkout. Subsequent config/source/operator/schema changes require a distinct reviewed signed policy; there is no automatic broad adoption. Never replace the queue or its records during this handoff.

Validation:

```
/home/const/subnet120-rewrite/.venv/bin/python -B -m unittest discover -s tests -p test_durable_audit_services.py -v
```

## Learner recovery

`ops.durable_learner_service` uses a separate durable ROOT-signed policy. It
checks the exact approved scientific source inventory, unchanged learner
configuration, source and training qualification approvals, qualification
translation, reward activation and original private authority identity.
Historical HEX learner approvals remain byte-for-byte unchanged; their encoding
is explicitly bound in the new signed policy. The durable policy itself uses
BASE64. A recovery startup requires the existing published controller and
committed persistent optimizer state; it cannot initialize a fresh lineage.
The unchanged scientific controller reobserves its original remote job ledger.
A private lifetime lock and inactive predecessor checks prevent a second owner.

Deployed units on Arbos.life:

- `affine-independent-nine-source-durable-api-v1.service`
- `affine-continuous-nine-source-durable-auditor-v1.service`
- `affine-ordinary-v3-durable-continuous-learner-v1.service`

All are enabled for the user default target, with `Restart=on-failure` and
15-second retry delay. User lingering is enabled. Controlled restarts of all
three passed using the same signed policies, preserving the SQLite queue and
single original epoch-29 remote training job. This is a service restart test,
not a physical reboot test. Existing GPU jobs, workers, checkpoints, optimizer
state, signed reports and hourly weight writer were not restarted or relabeled.

Fifteen focused tests cover signature and config/runtime drift, old admission
rows and cutoffs, queue inode/schema preservation, private identity, exclusive
locks, preserved historical approvals and recovery across an advancing epoch.
