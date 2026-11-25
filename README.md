# Affine SN120 v2: live-pool mining

**The official-active scoring pool rotates every round and MUST be refetched every round.** Start with the [v2 miner quickstart](docs/MINER_QUICKSTART.md) or the full [v2 miner guide](v2/miner/README.md). Use the official `v2` source, not legacy mining tools.

The authority is a fresh, cache-busted **https://affine.io/mining.json**, specifically **`round.pool.active` and `round.pool.active_sha`**. Fetch `https://affine.io/public/pool/active/<sha>.json` and require the SHA-256 of its bytes to equal `active_sha`. Use **https://affine.io/public/pool/summary.json only if the round has no pool**. The active object is a sorted, unique list of task IDs.

```bash
# From the repository root: resolve the current active IDs to verified task bodies.
python v2/miner/live_pool.py --out tasks.jsonl
```

Every custom HTTP request needs a `User-Agent`; cache only immutable SHA-addressed reserve shards. Base `dapo:`/`math:` bodies come from `v2/sn120_v2/envs/math_pool_v2_1.jsonl`, which is **STALE/test-only as an active list and NOT a king-draw source**. Generator-produced `gen:` and other reserve bodies come from **https://affine.io/public/pool/reserve/MANIFEST.jsonl** and `https://affine.io/public/pool/reserve/shards/<name>.<sha>.jsonl`; reserve rows provide `id`, `problem`, `answer` and optional public `prior_p`.

## Nine steps for a v2 miner

1. **Register an Ed25519 hotkey** on SN120 and run `python -m sn120_v2.links activate --wallet NAME --hotkey NAME --execute`. There is one entry per hotkey per round and no coldkey entry limit; a charged hotkey needs a new registration for another turn.
2. **Fetch the live pool again** with `python v2/miner/live_pool.py --out tasks.jsonl`.
3. **Generate rollouts and transform them to official batch JSON.** The reference `miner/kit/generate.py` defaults to `--pool live`; explicit offline/test operation is `--pool local --local-pool PATH`. Generator output is not an upload batch, and legacy `make_batch.py` compatibility with the live contract is not guaranteed.
4. **Run ordinary and live-pool checks.** First add `--no-live-pool` for the ordinary offline check, then omit it for the warning-only live diagnostic, or use `--active FILE` with a saved active object:

   ```bash
   python v2/miner/affine_miner.py check --mining-json https://affine.io/mining.json --hotkey <ss58> --no-live-pool batch.json
   python v2/miner/affine_miner.py check --mining-json https://affine.io/mining.json --hotkey <ss58> batch.json
   python v2/miner/affine_miner.py check --mining-json https://affine.io/mining.json --hotkey <ss58> --active active.json batch.json
   ```

5. **Open a sealed upload link and PUT** using the publisher's current commands:

   ```bash
   export AFFINE_PUT_URL="$(python -m sn120_v2.links open --hotkey <ss58> --key /private/miner.seed --links-url https://affine.io/ff/links)"
   python v2/miner/affine_miner.py upload --mining-json https://affine.io/mining.json --hotkey <ss58> batch.json
   unset AFFINE_PUT_URL
   ```

   The presigned URL is secret. See `mining.json.contract.upload` for the authoritative activation/open commands.
6. **Interpret file checks by phase.** IDLE means no round is open; waiting files are checked then and when a round takes entries. They are not checked while a round is RUNNING. Mid-round verdicts are the previous snapshot, identified by `queue.files_checked_at`, each row's `checked_at` and `batch_sha256`.
7. **Crown with strict `z > z_bar`.** The eligible highest-delta candidate must also satisfy the live symmetric gate and tie tolerance. The currently live value is **`z_bar = 0.5`**, but live params can change; derive it from `mining.json.params` as shown in the quickstart.
8. **Require fresh next-round paired confirmation before payment.** The crown is applied immediately, but only a successful new paired comparison in the next round confirms it; crown-round outcomes do not substitute for that evidence.
9. **Only confirmed registered kings split weight equally**, with no backpay at confirmation. One writer attempts weights on the hourly cadence in `params.weights_cadence_s`, preserves the last valid weights on chain/state failure, and never uses burn or owner-only fallback. Read **https://affine.io/public/winner-stats.json**; **https://affine.io/winner-stats.json** serves the same bytes.

## Rotation-v2 and draw settings are separate

With `rotation_rule = "info"`, candidate ranking uses **score = p(1-p)**. For sufficient history, `p` is the IRT-predicted head pass rate; otherwise it uses optional reserve `prior_p`, then configured `rotation_prior_p`. The maximum is **0.25 at p = 0.5**, and **score > 0.16 exactly means 0.2 < p < 0.8**. Scores from 0.15 through 0.16 may lie outside that interval.

**`params.irt_middle_band` belongs only to the separate `band` draw rule.** Its preferred interval is inclusive but nonexclusive; outside tasks retain sampling preference. It never governs rotation-v2 active-set ranking. See [pool rotation](v2/POOL_ROTATION.md) for the full algorithm and public replay evidence.

For actual verified head/master downloads, training, scoring, KL measurement and reproduction, use [v2/REPLICATE.md](v2/REPLICATE.md). The quickstart deliberately does not freeze batch caps or training knobs; the live params and `python v2/miner/affine_miner.py bounds --mining-json https://affine.io/mining.json` are authoritative.

**Privacy:** only crowned/applied winner batches and their training values become public. Loser, waiting and invalid batch bodies never become public. All timestamps are UTC.

Historical research and inactive designs (not the live miner protocol): [architecture and runtime](docs/ARCHITECTURE.md) ·
[MiMo operator lifecycle](docs/MIMO_LIFECYCLE.md) · [flip-incentive shadow scoring](docs/FLIP_INCENTIVE.md) ·
[fixed-nonce GRPO cutover plan](docs/MIMO_FIXED_NONCE_GRPO_CUTOVER_PLAN.md) ·
[retained-Adam NLL experiment](docs/EXPERIMENTAL_RETAINED_NLL.md) · [RCore adapter](docs/NATIVE_RCORE_COMMON.md) ·
[implementation plan](IMPLEMENTATION_PLAN.md) · [operational state and evidence](STATE.md) ·
[STANDING RULES](STANDING-RULES.md) · [live registration, payouts and rollback](docs/LIVE_SUBNET.md) · [changelog](CHANGELOG.md)

## v2 public task pool

The live public pool follows the active/reserve/retired lifecycle in the [pool rotation document](v2/POOL_ROTATION.md). When live params select `rotation_rule="info"`, `affine-pool-rotation-v2` ranks candidates by Bernoulli Fisher information and publishes the deterministic ranking and next-up queue. The public `/pool` page shows active-set composition, score histograms, recent admits/retires and per-task evidence. Always derive the active set from fresh live artifacts rather than checkout defaults; changes in this checkout do not deploy automatically.

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
