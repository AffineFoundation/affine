# Prospective MiMo evaluator and worker integration

This implementation is private and nonpayable. It does not activate the new
mining contract, publish public discovery, or change chain weights.

## Evaluation authority and semantics

`subnet.mimo_manifest.build(..., evaluation_sets=[...])` accepts signed suites:

```json
{"name":"fixed32","env_id":"affine_math","indices":[6746,6747],"seed":42,"repeats":1}
```

The example shows the shape, not approved task IDs. The operator must supply the
actual reserved split from the authenticated native snapshot. Each suite has
exactly these five fields. Indices must be distinct, inside that taskset, and
disjoint from its complete mining population. Suite names are unique; fixed32
may be a subset of heldout128. The signed taskset, template, sampling settings,
thinking mode, output horizon, final-channel policy, seed and repeats determine
the evaluation dataset ID. A change creates a different comparison series.

Evaluation samples every requested task and repeat. It does not require a mixed
success/failure group, search for a favorable nonce, filter all-correct groups,
construct proofs, or upload mining submissions. It uses the same cached MiMo
runtime, top-k/top-p implementation and final-channel native grader as mining.
Its separate deterministic draw domain keeps random draws fixed across
checkpoints for the same suite. Training and mining authority is never extended
to include held-out indices.

Per-attempt records retain the complete output token sequence, token count,
task and prompt hashes, EOS/budget stopping reason, native classification, reward,
and timing. Correct and completed wrong outputs are scored normally. Caps or
unfinished final channels remain unresolved. A grader/runtime failure is
indeterminate with a sanitized stage and exception type; it is never a fabricated
wrong answer. Any such failure makes the aggregate score null and the evaluation
status `error`, while preserving every attempted row for diagnosis. Lost worker
leases abort execution.

The signed selection is bounded to 1,250,000 output tokens per job and the queue
report to 12 MB. This accommodates fixed32 and heldout128 at 8,192 output tokens
with one repeat. Larger suites/repeat budgets require explicit smaller signed
suites, not silent dropped tasks or reduced horizons.

## Running an authenticated job

Use the regular signed worker envelope with `worker_policy` equal to
`mimo-complete-group-worker-v1`, `role` equal to `evaluate`, and:

```json
{"evaluation":{"version":"mimo-heldout-evaluation-v1","run_id":"original-run","set_name":"fixed32"}}
```

The envelope also pins the original signed opening, exact installed `subnet/`
Python inventory, package versions, job identity and deadline. Refreshed signed
checkpoint GET capabilities may be supplied as `checkpoint_read_urls`; they
do not change the immutable model descriptor or mining sampling draws.

For a single operator-owned job:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=1 \
  python -B -m subnet.mimo_evaluation JOB.json \
  --authority OPERATOR_PUBLIC_KEY --workspace /private/evaluator \
  --checkpoint-cache /private/authenticated-model
```

For the existing private role queue, register an `evaluate` worker and run
`subnet.distributed_worker --role evaluate` with the usual coordinator, signing
seed, authority and workspace options plus a bounded `--mimo-cache-bytes` value.
The seed stays on that worker. The persistent factory holds a checkpoint lease
through each evaluation and reuses the loaded model across jobs. A supplied local
checkpoint is copied into its managed cache and authenticated before use; an
already resident checkpoint is not downloaded or rehashed again per task.

The shared queue authenticates the report against the original suite, source,
runtime, checkpoint and ordered attempt coverage. The legacy evaluator and old
controller's legacy job builder remain separate: a MiMo opening alone does not
silently convert an old evaluation job into this new protocol.

## Training state, source versions and public cutover

The new train worker uses an explicit stable `--trainer-state-root`, outside
lease-attempt workspaces. It checks the live queue lease before optimizer,
publication and CURRENT changes. One named trainer owns the local optimizer
lineage. Model-only publication is streamed, immutable and fully read back;
optimizer state stays local. A lost acknowledgement resumes the original
transaction, while a crash after optimizer mutation but before complete staging
requires explicit recovery and cannot automatically run another update.

Two source identities are deliberately recorded. The original opening's
`source_bundle.sha256` identifies the miner release and is part of its sampling
challenge. The signed job's `source_files` identifies the actual reader/trainer
release. A compatible newer reader may process older original documents only
while all original runtime, harness, sampling and native-source checks pass.
The report binds both the original manifest digest and executed source inventory.
Never rewrite an old opening or claim a later release produced earlier samples.
Adding evaluation sets requires a new signed evaluation opening; it does not
replace the opening bound to existing miner commitments.

Public deployment still requires explicit work in these boundaries:

- `mimo_coordinator.Coordinator.open` permits private/test/nonpayable epochs and
  writes private opening/dispatch artifacts.
- `distributed_roles.Coordinator.enqueue` permits only `nonpayable-` epochs with
  `payable=False`.
- `mimo_training_worker.validate_job` requires a private epoch and publication
  namespace; `mimo_grpo.handover` advances a private local CURRENT only.
- Public discovery, next-epoch orchestration, miner registration/capability
  issuance, continuous audit/reward ingestion and dashboard run projection must
  use the new authenticated artifacts explicitly. These modules do not submit
  or replace network weights.

Before opening the new protocol, run the exact released source through real
native mining, independent proof verification, trainer recovery/model-only
publication, a baseline evaluation and evaluation of the updated checkpoint.
CPU tests establish dispatch and recovery behavior, not H200 numerical agreement
or a learning gain.
