# Affine epoch subnet

Bittensor subnet120 (MiMo GRPO). **Miners upload one batch file** (prompts, rollouts,
rewards) to a private per-hotkey object; at the miner's turn the validator copies it,
checks the format, trains one GRPO step from the published head and the promotion gate
keeps the candidate only if it improves the model on a private held-out set.
No mining client, no epochs or commitments, no TOPLOC, no proofs, no regrading.

- **[Mining: the batch file, upload and rules](docs/MINER_QUICKSTART.md)** ·
  reference helper [`miner/affine_batch.py`](miner/affine_batch.py) ·
  live contract https://affine.io/mining.json (`batch_file`) · https://affine.io/llms.txt
- Old-format (client) batches are trained until 2026-10-31 09:00 UTC (03:00 CR); the old
  client (`subnet.miner_supervisor`) has been removed.
- [Validator side of the free-file protocol (lifecycle, trainer v8.3.3h, activation/revert)](ops/free_file/README.md)
- [STANDING RULES (ops): nothing ever on the encrypted disk](STANDING-RULES.md)
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
