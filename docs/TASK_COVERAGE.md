# Manifest-driven task coverage

The coordinator can temporarily exclude recently overused training tasks from a
future epoch's signed `environments[].indices`. The existing public miner reads
that list for each new epoch. This does not require a new miner executable or a
different proof format. Custom clients must also use the current manifest;
persistently pinned `--indices` outside its list will be rejected.

This is a prospective coverage experiment. It does not establish that repeated
tasks caused the model's learning problems, that miners cheated, or that wider
coverage will improve held-out performance.

## Initial policy

- Start with the original 6,746 eligible mining tasks. Keep all held-out tasks out.
- Look at the preceding 12 finalized epochs. Count one exposure per task per
  completed optimizer update, using the actual reported training inputs.
- Temporarily exclude up to 256 tasks with the greatest exposure. Break ties
  deterministically using the epoch identity and task index.
- Preserve the two original calibration tasks, 1553 and 7121, and their positions.
- Keep the other 6,490 tasks in their original order. The full task registry stays
  available for historical verification; only this epoch's allowed list changes.
- Recompute at each new opening. The exclusion is temporary, not a permanent ban.

The initial implementation is default-off and supports a prospective activation
round. Its deployment status is documented in the current
[miner guide](../dashboard/public/llms.txt); the authenticated epoch manifest is
always authoritative.

K4/L4, eight distinct completed rollouts per batch, nine batches per UID, the
1,000-nonce search range, randomness verification, task grading, reward rules,
trainer objective and optimizer are unchanged. A cooldown is an eligibility
change, not a retrospective penalty. An earlier signed epoch keeps its rules.

## Evidence and restart behavior

The provider verifies the coordinator's signed closure, the original public
manifest, signed training job and signed input admissions. It selects a
successful original or recovery report only when its input and output
checkpoints match the completed epoch. It links reported training pairs to the
admitted batch contents and checks optimizer counters and checkpoint continuity.
The underlying trainer reports are unsigned operational evidence; those checks
do not turn them into cryptographic proofs of model execution.

The exact task selection, policy and evidence hashes are frozen in an
authority-signed local record before the coordinator signs the opening manifest.
Retries use that record. The binding excludes only the renewable source-download
URL, retaining its archive hash, size and every other contract field. A newly
available report cannot rerank an issued epoch.

Missing, corrupt or ambiguous history causes the next opening to retain the full
original pool, with a recorded fallback reason. Conflicts in an already frozen
selection are integrity errors and must not silently rewrite the epoch. Reads
have byte, candidate-count and elapsed-time limits. No verifier or model is run
to calculate this list. Training can still use a nonempty eligible set below its
512-task maximum; 512 is not a completion requirement.

## Why try this

In finalized epochs 128–151, 46,416 committed batches covered 3,120 distinct
tasks. The learner's 12,288 task-update exposures covered 2,758 of the 6,746
eligible tasks (40.9%). A supply-conditioned uniform-selection comparison
predicted 2,747 distinct trained tasks, close to the actual 2,758. That points to
the available submission pool as a useful place to investigate coverage.

A prior-history-only replay for epochs 140–151 found that removing the 256 most
trained tasks still left at least 700 distinct structurally captured tasks in
each epoch. This is observed-supply feasibility, not a forecast of how miners
will adapt or how many newly generated batches will pass native grading.

The broad remaining pool matters: same-epoch duplicates across miners still
earn zero new task points, so forcing everybody into a tiny shared task window
could increase collisions. The initial trial keeps 96.2% of the full pool open.

## What to measure

Track unique submitted, captured and actually trained tasks separately; new
coverage relative to the prior 12 epochs; subject and difficulty distribution;
external-miner participation and completed batch counts; native rejection and
duplicate rates; and full epoch duration. Compare held-out learning independently.
Do not count repeated task IDs as independent tasks or equate a structurally
captured batch with an audited-valid one. Reduce or disable cooling prospectively
if search yield or epoch continuity deteriorates.

A separate [prospective native-priority policy](NATIVE_TASK_PRIORITY.md) can prefer
submitted tasks missing from recent training during native admission. Cooling
controls availability; native priority controls the order of actual submissions.
Neither changes the miner proof format or reward rules.
