# Matched MiMo answer-format sensitivity (diagnostic only)

Both authenticated fixed32 evaluations were inspected on the same task indices,
repeat IDs, dataset and evaluation suite. Original signed reports and output
sequences were preserved. This is answer-grading sensitivity, not evidence of
cheating, model convergence, or correct reasoning.

| Interpretation applied to both evaluations | Baseline | First update |
| --- | ---: | ---: |
| Original signed v1 score (unchanged) | 18/32 | 19/32 |
| Single final box: optional terminal period and correct wrapper attachment | 22/32 | 22/32 |
| Additionally accept diagnostic task-appropriate trailing units | 22/32 | 23/32 |
| Additionally accept repeated identical native-positive boxes | 22/32 | 24/32 |
| Any native-positive final-channel box (unsafe diagnostic only) | 22/32 | 25/32 |

The narrow prospective parser was actually applied to all 64 original token
sequences and reproduced 22/32 for each checkpoint. Every v1-complete answer
retained its original grading text. There is therefore no demonstrated gain in
this matched narrow-format sensitivity, despite the official 18→19 difference.
A single update and 32 tasks do not establish a learning trend either way.
Do not compare updated relaxed scores with the original strict baseline.

## Evidence and difficult cases

Baseline neutral EOS answers:
- 6903: native-positive single naked box follows a *closed* inline-math expression
  and a blank line. v1 strips whitespace and misinterprets the previous closing
  dollar as an opening wrapper around the box. The initial diagnostic mirrored
  this bug; the explicitly derived v2 diagnostic corrects that category.
- 3689, 2077, 6873: native-positive single boxes followed by `$.`.
- 1849: invalid thinking-marker count; remains unresolved.

Updated neutral EOS answers:
- 437, 2077, 4414: native-positive single boxes followed by `$.`.
- 6498: native-positive `\\boxed{52}` followed by `$ inches.`. The unit fits this
  particular perimeter task. Arbitrarily dropping units is unsafe and excluded
  from the prospective parser.
- 335: three mathematically equivalent domain boxes. The original native grader
  rejects the set-builder box, but accepts the two interval-notation boxes.
  This is not three native-positive boxes. Selecting whichever box matches the
  reference would introduce answer-shopping; remains unresolved.
- 3574: two identical native-positive boxes. Also contains incorrect arithmetic
  in its reasoning, including `785/72 = 29/12`. Correct final answers do not
  validate reasoning. Repeated boxes remain unresolved prospectively.
- 4971: invalid thinking-marker count; remains unresolved.

Baseline has six budget-capped trajectories; updated has five. They are never
rescued by a box occurring before truncation. No thought-channel answer is used.

## Authentication and reproducibility

Baseline report SHA256:
`ce16a508f8743afe2799446660a548768e4bd425811ffd75af61d2cd71904439`

Updated report SHA256:
`d83134457b93ac970e404a62aa6ead35305bf6fe3003b97c3ca56f60c66a5e3b`

Tokenizer JSON SHA256 (checked against both signed checkpoint inventories):
`06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`

`diagnose.py` authenticates operator-signed jobs/manifests, worker-signed report
requests, report digests, exact source inventory and release policy; decodes the
original tokens; then invokes the frozen v7 original native environment grader
on each balanced final-channel box of strict-neutral EOS cases. The queue is
opened SQLite read-only. Native grading is local, without model generation or
training. `MATCHED-RESULTS.private.json` preserves the initial categorization;
`MATCHED-RESULTS-v2.private.json` records the explicit correction for task 6903,
with a hash binding to the first result. These include per-task classifications,
original output hashes, final texts and individual box verdicts.

`check_prospective.py` applies the isolated prospective parser to all original
64 token paths, confirms v1 behavior agrees with the frozen source, and checks
that every previously complete answer keeps identical grading text. New accepted
boxes are matched exactly to the original native grading evidence above.
`PROSPECTIVE-ACTUAL-TOKEN-PARITY.json` records this check and parser source hash.

## Narrow prospective change

A new explicit signed projection version accepts one balanced nonempty final
box, paired math delimiters and an optional terminal full stop. Delimiter state
prevents a preceding closed math expression from being attached to the box.
All EOS, control-token, thinking-channel and budget-cap requirements remain.
No units are stripped, no repeated/conflicting boxes selected, and no reasoning
error is silently relabeled as proven valid reasoning. Native answer grading is
unchanged. Historical v1 still has exactly its original semantics.

This changes the reward/completion grammar, so it needs a new qualified source,
signed release scope and epoch boundary, coordinated miner/native/audit/eval
rollout, and separately labeled matched evaluation baseline. It must not rewrite
old scores, reports or training inputs. No patch is deployed by this diagnostic.
