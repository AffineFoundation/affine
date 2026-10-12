# Periodic fresh TRAIN diagnostics

This operator worker generates new answers on one fixed cohort of 128 approved
MATH TRAIN tasks. It evaluates the original upstream reference, then promptly
evaluates the latest durable production checkpoint. Subsequent checkpoints need
at least four additional finalized optimizer updates. It coalesces a backlog to
the latest checkpoint and runs one checkpoint at a time on a dedicated
evaluation GPU. Production training does not wait for it.
The first reference attempt on October 12, 2026 stopped before model execution
because a local Python import shadowed an earlier runtime-version lookup. Its
failure evidence was retained. This revision fixes that import placement and
tests the actual entry path. The original upstream reference was subsequently
archived and published. The first completed current-checkpoint capture then
exposed an archive-reader collision: `task-exposure-metadata.json` shared the
raw task prefix. The reader now selects the exact signed 128 task filenames,
permits only that declared metadata companion, and still rejects missing,
changed or unexpected raw task records. Original generations are retained;
this correction does not rerun evaluation.

These are training-split diagnostics, not held-out confirmation, miner proof
verification, or a promotion rule. The cohort was selected without outcomes
from approved mining membership, excluding known evaluation identities. Fresh
generation does not imply that a task was absent from training.

## Interpretation

The initial authenticated exposure record through production step 67 contains
55 of the 128 tasks with at least one attributed update, with 73 not yet trained
in this run. Overall128 is TRAIN-split performance, not performance exclusively
on already-trained tasks. The fixed initial55 subset stays identical for the
reference and later checkpoints and is the primary training-fit curve. The public
API retains all128 counts and mean_reward; its broader score remains in the
tooltip. The chart requires complete, comparable, error-free all128 results,
uses the same55 mask at every checkpoint, and never falls back to all128 if the
mask is absent. A changed mask starts a separate series. Changing current exposure
groups remain descriptive.
Later checkpoints use their own closed training history. Missing or incompatible
attribution produces `unknown` exposure, never an assertion that a task was
untrained. Attribution
uses signed closure and job bindings plus local trainer report bytes; it is not
proof that physical execution occurred. Upstream pretraining is unknown.

The original upstream comparison checkpoint begins `2173b3e0`. It is distinct
from the current production run's actual genesis, which begins `6493a901`.
The reference's zero-update coordinate means zero updates in this diagnostic
comparison; it does not rename the production genesis.

Each checkpoint retains all 128 original cases, including wrong answers,
unresolved answers and infrastructure errors. The fixed profile uses one draw
per task, paired seeds, FP32 cached generation, temperature 0.7, top-p 1, and a
2,048-token output cap. The API `context_token_budget` field records the signed
profile ceiling of 8,192 tokens. The evaluator applies the smaller of that ceiling
and the checkpoint’s supported context: both initial checkpoints support 4,096,
so their actual limit is 4,096. Each prompt plus the full output budget must fit;
no prompt is truncated to satisfy this check.
Comparisons check actual prompt token IDs, task identity,
seeds, source and runtime profile. Caps, EOS and exposure groups are reported
separately. The already-trained subset can help measure fit; neither it nor the
overall TRAIN-pool score establishes unseen-task generalization.

## Operator entry point

Use Python 3.12 or later in the admitted operator environment:

```sh
python -B ops/training_diagnostics/monitor.py \
  --config /absolute/path/to/signed-monitor-config.json \
  --authority PUBLIC_ED25519_KEY_HEX
```

`--once` performs one monitor tick. The long-running form polls every 30 seconds.
Only one monitor may own the local queue lease. Run it under an operator-owned
supervisor with explicit resource and lifetime limits.

This directory publishes the exact reviewed operational source. It is not a
self-provisioning deployment. The operator supplies an independently reviewed,
signed configuration and base plan containing:

- Local queue and aggregate paths, read-only production state and run identity.
- The signed cohort, approved mining manifest, checkpoint inventories and fixed
  numerical profile, including exact source and runtime hashes.
- An authenticated dedicated host, SSH command, interpreter and lease expiry.
- Pinned hydration and closed-history attribution helpers, and the storage
  implementation and configuration used for full object readback.
- A separately protected signing key matching the admitted authority.

The monitor and CPU tests use PyNaCl; storage uses the configured
`subnet.storage.Bucket` implementation. Numerical execution requires the exact
signed PyTorch, Transformers, native grader and Affine source environment.
`evaluate.py` pins the public Affine authority: changing the CLI authority alone
does not turn this source into another authority's admitted evaluator.

Private deployment builders, configurations, capabilities, keys, host records,
task contents and evaluation outputs are deliberately absent. Supply those
through the operator environment, never through this repository. Copying these
files does not activate a service or authorize model execution.

## Original execution and preservation

`monitor.py` prepares signed assignments from genuine durable training closures.
`serial_worker.py` persists dispatch intent before starting the original process
and holds a serial lease across hydration and evaluation. An ambiguous original
dispatch is not reissued. The admitted configuration requires one attempt.

Plans expire within two hours, and the monitor refuses a new job with less than
two hours left on the dedicated host lease. Hydration and evaluation have
bounded subprocess waits. Operators must allow time for setup, capture and
storage readback within the actual rental lifetime.

`train_archive.py` validates the full 128-case inventory, downloads and checks
every uploaded original artifact in full, then publishes the signed aggregate.
Owned checkpoint-cache retirement requires its corresponding durable evidence.
The successful reference cache is retained in the normal completion path.
Failure holds also preserve complete model caches for their working lifetime;
the retirement helper explicitly refuses removal of the complete upstream
reference cache. Original durable models and evidence remain preserved.
Repeating an archive readback does not repeat numerical evaluation.

Any failed or abandoned original, infrastructure error, or comparability failure
holds the entire queue. Held queues report `healthy=false` with a fixed reason
code; a lease-blocked queue also reports unhealthy. Successful jobs continue the
four-update cadence automatically. There is no automatic numerical retry.

The subprocess wrapper does **not** guarantee termination of all native grader
descendants after a timeout. A failure hold prevents the next automatic model
dispatch; an operator must reconcile the original processes and perform reviewed
cleanup before reusing that host. The hold must not be cleared merely to advance
the queue.

## CPU controls

From this directory, with PyNaCl installed:

```sh
python -B -m unittest -v \
  test_training_eval test_training_lifecycle test_training_exposure \
  test_preflight_import test_completed_archive
```

The 30 CPU controls cover complete denominators, prompt and seed mismatch,
native unresolved classification, exposure uncertainty, four-update coalescing,
single-flight dispatch, crash intent, lease refusal, full-GET-before-publication,
protected reference retention and failure holds. The completed-pair controls
exercise latest-checkpoint publication after a baseline archive containing
exposure metadata, and reject missing or unexpected numbered raw records.
The actual evaluator entry test reaches the runtime-version check without importing a model. Tests make no
model or network calls. The signing key in the fixtures is explicitly synthetic.
