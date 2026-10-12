# Separate positive and negative training diagnostics

The trainer now records each selected pair's positive and negative sequence
scores before and after its update. A better preference margin can result from
raising the positive score, lowering the negative score, or both. The margin
alone cannot identify which happened. These fields make that distinction
observable without changing the training objective.

The logging-only handover is active for future training jobs from epoch 166,
following the authenticated learner transition on October 12. The current model
and Adam state were retained at optimizer step 75. At activation, epoch 166 was
still collecting and its first component-bearing training report had not yet
completed. Publishing code alone does not activate a release: the learner uses
its separately authenticated source declaration. Historical jobs retain their
original sources, reports, model checkpoints and optimizer lineage.

## Fields and interpretation

The original trainer report contains
`training.persistent_diagnostics.sequence_components`:

| Field | Meaning |
| --- | --- |
| `status` | `complete`, or `unavailable` with a fixed reason code |
| `pair_count`, `rows` | Every selected pair, in its original order |
| `pair_index`, `pair_sha256`, `task_sha256` | Bind each row to the existing training identities |
| `positive`, `negative` | Output-token count, prompt-token count, turn count and last turn's recorded `done` value |
| `before`, `after` | Positive/negative total and mean output log probabilities, plus the original raw pair margin |
| `delta` | After minus before for both sequence scores and the raw margin |
| `total_gradient_weight`, `gradient_observations` | References to the existing update's pair weights and use counts |
| `task_weighted_delta` | Average pairs within each task, then average tasks |

Log probabilities use natural logarithms. The production sequence score is the
**sum of generated-token log probabilities divided by the output-token count**,
across all turns. Prompt tokens are not scored. The raw margin is positive mean
log probability minus negative mean log probability; it is not multiplied by
the preference scale beta. Total log probabilities are diagnostic fields only.

A positive `delta.positive_mean_logprob` means the positive trajectory became
more likely per token under this update. A negative value for the negative
trajectory means it became less likely. A rising margin does not imply the
positive trajectory improved, and fitting these selected trajectories does
not establish better fresh-sample or held-out accuracy. Token counts describe
the actual optimizer inputs; the recorded `done` field is not by itself a
new EOS-versus-budget classification or a change to completed-answer grading.

`input_checkpoint` and the optimizer counters are explicit. The output checkpoint
is the containing original trainer report's `new_checkpoint.id`. The existing
job hash, manifest and finalized completion bind the report to the run. There
are no invented retrospective component values for older epochs.

## Execution and bounded reports

The recorder observes the already-computed scalar sum and mean in the existing
reference and post-update forward passes. It detaches those scalars and retains
no tensor or gradient graph. Recording is inactive during gradient accumulation;
it adds zero model forward passes and does not change losses, gradients, Adam
moments, learning rate, precision, task selection or source eligibility.

The report is capped at 65,536 pairs and 64 MiB. Incomplete, nonfinite, malformed or
oversized diagnostic data produces `status: unavailable` with a fixed reason
code; it does not invalidate an otherwise successful update. The report adds
no rollout token arrays, full-vocabulary arrays, credentials, arbitrary exception
text or new upload capabilities. Existing input-assurance labels still apply:
recording a trajectory's likelihood does not prove that its reasoning is correct
or upgrade an unaudited input into a verified one.

The qualified H200 comparison used four pairs and two private updates per arm.
The original and instrumented runs had identical gradients, master parameters,
Adam moments, counters, exported BF16 hashes and forward counts. Four pairs added
5,178 diagnostic bytes, approximately 2.15 MB when extrapolated to 2,048 pairs.
That small fixed-order comparison qualifies equivalence for the tested execution;
it does not establish a speedup or a training-quality improvement.

## Retention and publication

Preserve these fields in the original completed trainer report and its existing
durable history record. Keep the original report hash and input/output checkpoint
bindings when exposing it through finalized history. Never reconstruct component
scores from the aggregate margin, overwrite historical reports, or publish
open-epoch submissions to fill missing diagnostics. Local replica retirement
continues to require the existing durable-evidence checks; this logging change
does not create a new cleanup path.

CPU controls compare the exact qualified pre-instrumentation AST with the logging
version and execute both paths. They check FP32/BF16 losses, gradients, Adam
moments, parameter updates and forward counts, the optional positive-NLL branch,
bounded reports, and fail-open diagnostic errors. Release-admission controls
reject forged evidence, source substitutions, changed optimizer lineage and
recursive logging predecessors. Existing historical release tests remain valid.

```sh
python -B -m unittest discover -s tests -p 'test_sequence_component_*.py'
python -B -m unittest discover -s tests -p 'test_unaudited_execution_*.py'
```
