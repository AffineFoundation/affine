# Completed-answer MATH cutover and base restart

Status, 2026-10-08: the base-model and optimizer reset has been applied at
global epoch 78. The API, learner and six reachable verifier workers have the
completed-answer source admitted. The first new manifest remains behind the
normal checkpoint calibration/opening checks; no completed new training epoch
is claimed by this record. An independent, fresh fixed-32 evaluation is running.

## Outcome contract

An epoch opts in by publishing an `affine_math` spec with adapter `prime_v1`,
`max_turns: 1`, version `prime-v1-2-completed-math`, and the config entry
`math_outcome_policy: completed-boxed-math-outcome-v1`.

Only responses with a nonempty, balanced final `\boxed{...}` answer are sent
to the existing native mathematical grader. A missing, empty, or unfinished
latest boxed expression is **unresolved**, whether generation stopped at EOS
or at the output budget. Unresolved attempts satisfy neither sample quota and
cannot enter training. A complete answer at the output budget is still graded;
the stopping condition alone never makes an answer incorrect.

The miner continues searching after unresolved attempts. Signed learner-document
admission rejects unfinished displayed answers, while native eligibility
independently decodes the committed tokens and applies the same rule. Display
text cannot override token-derived eligibility. Sampling verification remains
separate from outcome grading; valid unaudited submissions still need not wait
for inference audits to become training candidates.

Unmarked historical specs retain their original interpretation. Existing
submissions must not be relabeled or penalized retroactively for this change.
The new environment hash includes the completion helper and adapter bytes.

## Requested restart

Use the original `Qwen/Qwen2.5-Math-7B-Instruct` model at revision
`ef9926d75ab1d54532f6a30dd5e760355eb9aa4d`, not a learned checkpoint from the
current run. The original published checkpoint ID is
`6493a901bd009f0800d5eed97d19aee78947cc586afeb27abfb2c72032ad1924`.

The cutover must preserve old checkpoints, epoch history, audit reports, and
scores. It starts a separately authorized optimizer genesis with zero prior
steps, plus a fresh evaluation baseline on the unchanged held-out split.
Changing model weights alone is not an optimizer reset. Loss, learning rate,
task quotas, sampling randomness, and scoring are separate choices and must
not change implicitly with this restart.

Publish the grading-compatible source and miner instructions before opening
the first new epoch. Miners already follow checkpoint changes through the epoch
manifest; the model reset uses that handover. This grading-contract update also
requires compatible client code, not just downloading different weights.

Activation requires a qualified successor source/runtime, authenticated base
checkpoint, completed prior-epoch boundary, explicit genesis admission, and
updated controller/verifier/trainer source approvals. A recovery service that
requires the old optimizer lineage must not be used to silently initialize
the fresh run. Confirm the first published manifest, accepted completed-answer
batches, fresh optimizer lineage, training completion, and held-out evaluation
before reporting the restart as live.

## Run display and operations

affine.io retains its existing layout. An authenticated run boundary filters
the learning charts to this run and displays its first global epoch (78) as
epoch 1. Historical database rows and incentive records remain intact. Fresh
base-model evaluation records cannot attach to the ancient first use of the
same checkpoint; their state is explicitly scoped to this restart.

Training blacklist eligibility is refreshed by a read-only assessment process
that authenticates the same original audit evidence and writer policy. It
never submits a chain transaction or changes the writer cursor. This separates
the learning-loop opening from an unresolved historical payout transaction.
The hourly weight writer still needs actual-chain reconciliation of that
transaction; the reset does not claim to have resolved it.

The independent evaluation preserves the original 32-task cohort, seeds,
1024-token diagnostic budget and native grader. A new baseline starts at
optimizer zero and subsequent genuinely committed checkpoints are evaluated
without requiring ten updates first. This diagnostic is smaller than the
750-task held-out population and does not establish convergence alone.
