# Affine epoch subnet

Synchronous inference-verification training loop for Bittensor subnet120: publish
checkpoint → private miner batches → freeze/public audit → verify/score → train →
next checkpoint. Blockchain-free mock and live chain adapters share the same core.

- [Miner pilot instructions](docs/MINER_QUICKSTART.md)

> Batch cap notice (2026-10-30): the live per-miner cap is **125 tasks x 8 rollouts = 1,000 rollouts** per K=1 turn, and bigger batches (up to 125 tasks) are encouraged. To get it, run the v8.2.3u client in extended mode: the miner supervisor runs the approved v8.2.3u source automatically (https://affine.io/releases/mimo-source-v8.2.3u/README.txt); do not pass `--no-upgrade`, and omit `--max-batches` (never set it to 9). Extended mode starts from the next opening after your first v8.2.3u commitment. v8.2.2 clients and `--no-upgrade` are legacy and limited to 9 tasks. The signed manifest's top-level `max_batches` (9) is that legacy contract; the extended cap is `dual_accept_policy.extended_max_batches` (125) and the live value is https://affine.io/mining.json -> `training.batch_cap.max_tasks_per_miner`. https://affine.io/llms.txt is authoritative.

- [Flip-incentive SHADOW scoring (not LIVE weights)](docs/FLIP_INCENTIVE.md)
- [Architecture and runtime](docs/ARCHITECTURE.md)
- [Prepared MiMo fixed-nonce GRPO cutover (not active)](docs/MIMO_FIXED_NONCE_GRPO_CUTOVER_PLAN.md)
- [MiMo miner migration and runtime requirements](docs/MIMO_MINER_MIGRATION.md)
- [MiMo operator lifecycle and independent audit loop](docs/MIMO_LIFECYCLE.md)
- [Experimental retained-Adam NLL objective (inactive)](docs/EXPERIMENTAL_RETAINED_NLL.md)
- [Live registration, payouts and rollback](docs/LIVE_SUBNET.md)
- [Implementation plan](IMPLEMENTATION_PLAN.md)
- [Operational state and evidence](STATE.md)
- [Original RCore common adapter and private terminal package](docs/NATIVE_RCORE_COMMON.md)
- [Changelog](CHANGELOG.md)

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

Isolated signed GPU role controls can be inspected with:

```bash
.venv/bin/python -m ops.check_gpu_role_evidence --state /private/operator-gpu-control-state
```

This binds signed jobs to operator-collected reports, verifies frozen R2 bytes and
checkpoint handover, and requires an independent published checkpoint byte audit.
It does not itself recompute inference or prove continuous controller integration.
