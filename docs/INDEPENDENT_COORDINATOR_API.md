# Independent verifier queue API

The learner currently starts the verifier HTTP API inside its own process.
Model/source initialization delays and learner recovery can therefore interrupt
otherwise independent continuous audits. The optional standalone service keeps
that API alive while the learner restarts.

This is prospective, default-off code. It is not part of the sealed FE source
or the current deployed queue service. Activation requires a reviewed source
contract, matching role identities/lease configuration, and an explicit handover.

Set `remote.verifier_queue.external_api=true` in the operator configuration.
The learner then opens the existing queue database without binding or stopping
its HTTP server. Run the independent service on the same Arbos.life host:

```sh
python -m subnet.coordinator_api_service --config PRIVATE_CONFIG \
  --authority-seed PRIVATE_SEED_PATH --expected-authority PUBLIC_AUTHORITY_KEY
```

The seed must remain private and local. Give the standalone service normal
process supervision, independently of the learner. It uses the identical
`state/roles/verifier-queue.sqlite3` database, worker signatures, report schema,
atomic claims, expiration and retry policy. Historical signer trust is retained;
inactive historical workers cannot claim new jobs. It does not fetch models,
change audit draws or admit new scientific source contracts.

Before activating, stop the old embedded API at a verified boundary and start
exactly one API listener with the same address and queue configuration. Keep
both processes on the same local filesystem; do not put SQLite on R2 or mount
it across GPU machines. GPU workers continue accessing only the authenticated
HTTP API. No queue deletion, job redraw or lease reset is needed.

Controls verify that a claim survives learner connection recreation and that
the original signed report can complete through the independent connection.
Existing queue controls cover concurrent atomic claims and stale lease rejection.
