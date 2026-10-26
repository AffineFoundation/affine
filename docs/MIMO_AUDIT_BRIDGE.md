# Prepared MiMo audit and hourly-assessment integration

This code is prepared for the new MiMo release. It does not activate a public
contract, install a service, replace the existing hourly writer, or change any
historical assessment. Public execution additionally requires the signed
qualified release policy and the operator's local public opt-in.

## Population and continuous audits

`mimo_audit_bridge.build_population` authenticates the original signed opening,
registration snapshot, structural captures, miner commitments and native
dispatch. Its `committed_at` must equal the coordinator's original
`POPULATION.json.sealed_at`; `round` continues the existing network sequence even
when the model run restarts. Obtain each original commitment from the captured
immutable snapshot, checking its recorded digest. Do not convert these documents
to the old `commitment.batches` format.

The resulting population retains every committed group for auditing. Reward
eligibility includes unique native-eligible groups excluded only by the
trainer's compute budget. It excludes native-invalid and ineligible groups and
applies the existing cross-miner task-collision rule. Eight trajectories still
count as one task contribution. Neither training completion nor a sampling audit
is required to record native eligibility.

Sign and persist that population, then call `AuditScheduler.admit`. The scheduler
uses the existing `distributed_roles.Coordinator` SQLite queue, registered
verifier identities, and new MiMo worker jobs. `tick` samples randomly from
current and historical committed groups. It writes the random selection before
capability creation and dispatch. Lost enqueue acknowledgements resume the
original job; expired or failed execution can obtain a bounded new job for the
same selected contribution. Failed execution is not a cheating verdict.

The scheduler's metadata callback must return the admitted worker `source_files`
and `runtime_versions`, plus refreshed `checkpoint_read_urls` if necessary.
Source admission is distinct from the original miner source archive. The
original signed opening is never rewritten to renew a download URL. Immutable
proof slots stay private and use scoped GET capabilities.

`admit_queue_reports` requires the actual completed queue row, original
operator-signed job, original worker-signed report request and queue lease token.
It checks the qualified release, exact execution source and runtime, original
commitment and all eight nonce/token-coverage results. It emits the existing
typed observation schema only after those new-format checks. Repeated delivery
does not create another observation. Invalid worker evidence is refused with no
fraud claim; only an authenticated definite invalid-sample result produces a
penalty observation. The existing exact fallback remains part of the verifier;
no new numeric tolerance is introduced here.

## Existing writer composition

The existing deployed writer already accepts
`run_once(..., evidence_loader=...)`. Keep its locks, cursor, six-hour EMA,
last-best fallback, current-registration mapping and commit/reveal transaction
code unchanged. A separately reviewed, signed operator wrapper can provide:

```python
new_source = EvidenceSource(
    signed_reader_policy, authority,
    snapshot=existing_snapshot,
    current_estimates=existing_current_estimates,
    observation_reader=existing_observations,
    allow_public=explicit_local_public_opt_in,
)
loader = compose_loader(existing_load_evidence, new_source)
existing_writer.run_once(..., evidence_loader=loader)
```

The three reducer callbacks must be the actual modules pinned by the writer's
signed source policy. The reader policy authenticates its release envelope,
directory, existing queue path, execution registry, audit policy and verifier
roster. The reader uses a read-only SQLite connection and does not construct a
queue or signer. It adds MiMo evidence hashes without replacing the legacy
source-admission or numerical-policy hashes that the unchanged writer validates.

During overlap, the explicit `conservative-overlap-minimum-factors-v1` policy
retains both contribution histories and the stricter validity, resolution,
penalty and blacklist factors for each overlapping miner. This is a conservative
transition rule, not a new combined Bayesian posterior. Old evidence continues
to age under the original reader. An unavailable or corrupt new source raises
into the existing last-best fallback; it must never be replaced by an empty
success result. No part of this adapter substitutes owner-only or burn weights.

## Remaining activation requirements

The release lifecycle must publish the signed population and run this scheduler
independently of training. The operator must pin and install the new read-only
writer composition while preserving the sole writer and original cursor. Those
are future activation steps; they have not been performed by these modules.
Actual qualified GPU/native results, exact release receipts, public registration
and discovery handover, and the genuine next-checkpoint evaluation remain
separate release gates. The CPU tests cover real compact transport/capture,
Ed25519 queue requests, restart/ACK handling and evidence admission with explicit
local model/native fixtures; they do not qualify numerical behavior or learning.
