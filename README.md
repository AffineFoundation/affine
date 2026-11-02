# Affine epoch subnet

Bittensor subnet 120 (MiMo GRPO). **Miners upload one free batch file** (prompts, rollouts,
rewards) to a private per-hotkey object; at the miner's turn the validator copies it,
checks the format, trains one GRPO step from the published head (trainer v8.3.3l) and the
promotion gate keeps the candidate only if it improves the model on a private held-out set.
Rewards are used as given; no mining software is needed.

**Gate (prior-v1 baseline):** GREEN iff gain > 0, z > 1 (flat bar, no time relaxation),
pooled cap-hit < 0.33 and at least 100 tasks scored. GREEN -> the candidate becomes the head and
the turn earns; RED -> the head is kept and the turn earns nothing.
**GREEN files are public** byte-exact at `https://affine.io/k1/turns/<epoch>/batch.json`
(with their lr / kl in `turn.json`); RED files and RED training params stay private.

- **[Mining: the batch file, upload and rules](docs/MINER_QUICKSTART.md)** ·
  reference helper [`miner/affine_batch.py`](miner/affine_batch.py) · [miner kit and validator code](miner/README.md) ·
  live contract https://affine.io/mining.json (`batch_file`) · https://affine.io/llms.txt
- Optional per-turn `lr` (1.25e-7 to 1e-6, default 5e-7) and `kl` (0.005 to 0.1, default 0.02) in an `affine-batch-v2` file: [format and math](docs/MINER_QUICKSTART.md#optional-affine-batch-v2-optional-lr-and-kl-from-2026-10-31) · [validator side](ops/ff_train_params/README.md)
- Optional per-turn `step_mult` (integer: 1, 2, default 1; duongna's step multiplier) and optimizer / loss knobs (per-group lr scales, adam_eps, weight_decay, clamp_mode, loss_agg, adv_center, adv_clip; all default to the previous trainer) in an `affine-batch-v2` file, from e575: [fields and math](docs/MINER_QUICKSTART.md) · [validator side (trainer v8.3.3l, replay, activation/revert)](ops/step_mult/README.md)
- [Held-out gate and prior-v1 baseline (code)](priors/gate/README.md)
- [GREEN-file publishing (default on)](ops/green_publish_default/README.md)
- [Validator side of the free-file protocol (lifecycle, activation/revert)](ops/free_file/README.md)
- [STANDING RULES (ops): nothing ever on the encrypted disk](STANDING-RULES.md)
- [Live registration, payouts and rollback](docs/LIVE_SUBNET.md)
- [Changelog](CHANGELOG.md)

History and inactive designs (not the live protocol): [architecture and runtime](docs/ARCHITECTURE.md) ·
[MiMo operator lifecycle](docs/MIMO_LIFECYCLE.md) · [flip-incentive shadow scoring](docs/FLIP_INCENTIVE.md) ·
[fixed-nonce GRPO cutover plan](docs/MIMO_FIXED_NONCE_GRPO_CUTOVER_PLAN.md) ·
[retained-Adam NLL experiment](docs/EXPERIMENTAL_RETAINED_NLL.md) · [RCore adapter](docs/NATIVE_RCORE_COMMON.md) ·
[implementation plan](IMPLEMENTATION_PLAN.md) · [operational state and evidence](STATE.md)

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
