# Affine epoch subnet

Synchronous inference-verification training loop for Bittensor subnet120: publish
checkpoint → private miner batches → freeze/public audit → verify/score → train →
next checkpoint. Blockchain-free mock and live chain adapters share the same core.

- [Miner pilot instructions](docs/MINER_QUICKSTART.md)
- [Architecture and runtime](docs/ARCHITECTURE.md)
- [Live registration, payouts and rollback](docs/LIVE_SUBNET.md)
- [Implementation plan](IMPLEMENTATION_PLAN.md)
- [Operational state and evidence](STATE.md)

Run `.venv/bin/python -m subnet.mock` for a real model/R2/training smoke, or
`.venv/bin/python -m subnet.tests` and `.venv/bin/python -m ops.new_subnet_checks`
for protocol checks. Credentials and generated artifacts stay outside Git.

Read AGENTS.md before modifying this repo. Legacy production and its private archive
remain separate; this checkout does not automatically deploy changes.

Independent, read-only evidence checks for retained experiments:

```bash
.venv/bin/python -m ops.check_epoch_evidence
.venv/bin/python -m ops.check_service_evidence
```

The first checks completed multi-environment research epochs. The second checks
the continuous controller's deadline, signed frozen artifacts, independently
accepted batches, changed weight bytes, comparable held-out evaluations and the
next epoch's checkpoint binding. Both reject payable test records and confirm
their scores are excluded from payout aggregation; neither submits chain writes.
Authority-scoped checkpoint descriptors preserve independent controller histories.
Missing after-training evaluation files remain pending rather than counting as
completed epochs. These checks establish the recorded runs, not general model
improvement or completion of all environment integrations.
