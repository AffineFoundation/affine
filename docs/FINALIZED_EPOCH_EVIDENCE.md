# Finalized epoch evidence

The read-only evidence exporter publishes the retained learning record for
finalized epochs from round 14 onward. It does not change training, sampling,
mining admission, rewards, or the dashboard layout.

Start at **https://affine.io/history.json**. This is an Ed25519 envelope with
`payload`, `signer`, and base64 `signature`. The expected authority is
`3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd`.
Verify the signature over UTF-8 JSON of `payload`, with sorted keys, compact
separators, and no nonfinite numbers, using the same canonical representation
as `subnet.storage.canonical`.

The additive `payload.finalized_epoch_evidence.catalog` descriptor gives a
GET-only presigned URL, SHA-256, byte size, and expiry for the catalog. Existing
legacy history fields remain intact. The export also maintains an independent
signed index at the bucket key
`public/streams/nonpayable-live-reward-math-v1-/finalized-evidence/history.json`;
legacy history refreshes cannot erase this independent index.

## Reading the artifacts

1. Authenticate the history envelope against the authority above.
2. Download its catalog using GET and verify the **downloaded bytes** against
   the descriptor's SHA-256 and size.
3. Select an epoch from `catalog.epochs`. Each row binds its round, run label,
   input/output checkpoints, completion time, and signed source-record hashes.
4. Download the desired category from `row.artifacts`, again checking byte
   size and SHA-256. The `.json.gz` hash covers the **compressed bytes**.
   Decompress only after checking that hash, observing `uncompressed_size`.
5. Read the category's `status` and provenance before interpreting its data.
   An unavailable record is not a zero measurement.

URLs expire. Refresh the signed history to obtain current URLs; never treat
an expired capability as evidence that the epoch did not occur. No bucket
credentials are needed. These capabilities grant reads of exact public
artifacts, not writes or directory listing.

## Categories

| Category | Retained evidence |
| --- | --- |
| `training` | Each actual update's loss components, pre-clip gradient norm, learning rate and optimizer hyperparameters, accumulation precision, task/pair counts, and input/output checkpoints. |
| `training_inputs` | Compact actual gradient inputs: full prompt/output token IDs and retained output text, attributed to signed admissions and batch hashes. Pair hashes join these rollouts to the individual update records. Full-vocabulary arrays and proof tensors are omitted. |
| `exclusions` | Authenticated collection/admission decisions, including reasons and selection limits where retained. Reward collision exclusions and training admission are separate decisions. |
| `log_metrics` | Authenticated structured trainer diagnostics plus retained worker-log events and loading/saving progress, with explicit availability/provenance. |
| `evaluations` | Retained fixed32 and heldout128 per-task verdicts, lengths and stopping information, with exact checkpoint, cohort, harness, and source bindings. |

Training input counts distinguish admitted batches, actually trained tasks,
unique pairs, and repeated pair exposures. A recovery update follows its signed
successful training record, rather than a superseded failed job. A closure
without an authenticated training update does not acquire invented losses or
rollouts.

Evaluation association is by checkpoint: `input_checkpoint` and `post_update`
do not claim that the evaluation ran during that epoch. Cohorts and token
budgets can differ; compare only matching evaluation protocols. Historical
reports sometimes retain counts and hashes without the output text or final
token. In those cases a cap-length response cannot be distinguished as EOS
versus budget exhaustion, and the export reports that limitation. Recorded
native incomplete outcomes remain distinct from infrastructure errors.

Some early evaluation summaries authenticate aggregate scores and task hashes
without authenticating each original per-task verdict assignment. Their retained
unsigned reports are explicitly labeled that way. An unsigned failed-job
observation does not create task outcomes or a zero score. The sanitized retained
worker logs are under `log_metrics.payload.runtime_log`; these are operator-read
logs, separately labeled from the signed training diagnostics. Known progress
lines become numeric observations; blank and unrecognized line counts remain
visible. Raw private log text is not published.

The existing retained-Adam endpoint study's original128 and exposed_test128
records are labeled separately from ordinary diagnostics. Publication neither
asserts their historical independence nor claims an improvement before the
predeclared comparison is complete. Prospective private research tasksets are
not discovered or exported.

## Publication boundary

Only a matching, authenticated original manifest and durable learner completion
admit an epoch. A score file, a training job's existence, or an open submission
does not. Run labels distinguish old rounds 14–77, the completed-math restart
78–90, and the fp32-conservative run from 91 onward. Rounds that never finalized
are not fabricated as completed training epochs.

`ops.publish_finalized_epoch_evidence` reads the existing records and stages
allowlisted projections outside live controller state. Its default is a local
prepare-only operation; `--publish` uploads content-addressed projections,
checks their durable bytes, signs the index, and atomically exports the static
history file. An independent timer refreshes finalized evidence and read URLs.
It does not launch training/evaluations or alter live source contracts.

Secrets, credentials, upload capabilities, full-vocabulary arrays, raw private
paths and unreviewed free-form logs are not exported. Missing or unauthenticated
historical evidence remains explicitly unavailable.
