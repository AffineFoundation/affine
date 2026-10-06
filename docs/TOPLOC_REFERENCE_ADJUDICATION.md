# Independent TOPLOC reference diagnostics

Epoch 29 has genuine signed audit reports rejecting committed artifacts with
`InvalidSample: TOPLOC`. These reports authenticate the executed model/runtime,
source, checkpoint and artifact, but record neither the failing segment nor its
exponent/mantissa error measurements. A rejection alone does not establish intent
or distinguish corrupted proof data from cross-node numerical disagreement.

`ops/toploc_reference_adjudication.py` is an offline research diagnostic, not a
production verifier or reward policy. Run it on a second independently qualified
SM90 node, in a fresh process, with the **original frozen source** and checkpoint.
ROOT must approve the whole source bundle, node/runtime and artifact selection.
It checks exact torch/transformers/TOPLOC package versions before loading the GPU
runtime. It accepts the original signed job and signed worker report request, verifies
bindings, the complete committed task tuple and original selected-child identity,
and the immutable artifact digest, verifies all executed source pins,
and constructs the unchanged original GPU runtime. It records native TOPLOC
metrics by wrapping the instance's proof comparison without changing its result,
thresholds, sampling check or grader. Existing local corpus assets are required.
It downloads no model, uses no credentials, dispatches no job, signs nothing and
writes only a new explicitly research-only output file. Never import its output
as an accepted production report.

Example after ROOT supplies locally staged inputs (paths are illustrative):

```bash
python -B ops/toploc_reference_adjudication.py \
  --job original-job.json --report-request original-worker-request.json \
  --authority APPROVED_AUTHORITY --worker ORIGINAL_WORKER --child CHILD_INDEX \
  --artifact original-committed-artifact.zip --source-root frozen-source \
  --checkpoint original-checkpoint --asset-root pinned-corpus-assets \
  --output new-reference-research.json
```

For each of the two rollouts, collect every proof segment's exponent mismatches,
mean/median mantissa errors, proof hash, and the unchanged full-verifier result.
The runtime checks local model file hashes before loading weights. Repeat on the
original worker when useful; include an honest control and a deliberately
modified proof control under the same runtime. ROOT should compare node identity,
software/hardware profiles and measurements before deciding whether a production
reference adjudication is justified.

Interpret conservatively:

* An independent pass conflicting with the original rejection warrants a numeric
  investigation; it cannot silently rewrite the original signed result.
* Repeatable rejection under the same approved runtime supports invalidity under
  that contract. It still does not identify deliberate cheating.
* Reference infrastructure errors and numeric ambiguities return `reference_valid: null`
  with a distinct classification; disagreements remain unresolved. Never
  manufacture success, increase tolerances automatically or call these fraud.

The current authenticated adjudication contract only resolves an original
`numerical_ambiguous` observation. These original TOPLOC observations were labeled
`confirmed_invalid`, so conflicting research passes **cannot** be inserted into
that existing contract. Any correction requires a separately reviewed, explicit
resolution policy binding the original and independently authenticated reference
reports, while preserving every original record. For future contracts, record
TOPLOC mismatch metrics and route suspected numeric disagreement to bounded
reference adjudication before a confirmed-invalid penalty. This document and
research tool do not activate such a policy.
