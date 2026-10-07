# Prospective fixed-stream quota study

This additive research preparation is **not dispatchable or deployed**.
[The machine-readable plan](../research/paired-quota-cp24-plan.json) names CP24,
its Adam step-24 descriptor, frozen f213 scientific source and the same eight
predeclared tasks used by the separate negative-termination control. The older
interrupted research execution is not resumed, relabeled or treated as a result.
A newly authorized common-stream study must have distinct scopes and handles.

## Generation and sample preservation

Admit the exact CP24 model, all original optimizer-state shards, scientific
runtime/package inventories and calibrated inference/native verifier first.
Keep the native autoregressive harness, temperature 0.8, top-p 1 and cap 1,024.
Freeze one fresh research epoch/context before observing results; both quota
arms use that context and the exact attempt IDs 0 through 15 for each task.
The quota comparison does not change sampler draws, termination, learning rate
or reference anchoring.

`ops/paired_quota_common_stream.py` is default-off and calls the existing
runtime's `rollout` and `verify` methods. It collects **all 16 attempts**, even if
K2 completes sooner. The artifact sink receives the original rollout together
with probability arrays and TOPLOC proofs before any metadata projection.
Each attempt has a durable status row; missing/native/runtime errors, numerical
unknowns and confirmed verifier rejections do not become legitimate failures.
The stream preserves those records and all generated artifacts. Sink failure
aborts rather than silently repeating a seed or changing context.

The helper requires a separately supplied runtime-admission callback and durable
artifact/metadata sinks. It does not itself authenticate loaded model hashes,
operator signatures or bucket readbacks. CPU tests use synthetic admitted
runtimes, arrays and sinks; they are not real GPU proof qualification.

## Nested selections and matched updates

Select the first distinct verified successes and failures in attempt order.
K1's one success and failure are members of K2's two successes and failures.
Repeated content from another attempt does not fill quota. Never reuse one
member in two pairs or form the four-pair cross product. `selected_artifact_refs`
connects each selected execution/content identity back to the original archived
rollout digest and full probability/proof artifact reference; it performs no
readback or proof authentication itself.

Retain a supply report for all eight predeclared tasks, including completion
prefixes and unavailable/duplicate counts. A task unable to fill K2 under the
common budget enters neither **matched** training arm; do not replace it with an
easier task or retry new randomness. Report K1 coverage of excluded tasks too,
since this matching condition biases the compared task population and does not
measure population-wide throughput by itself. If no task qualifies, neither
arm trains and the study reports an unmet quota, not a failed learning result.

Restore both branches independently from the same CP24 weights and full Adam24
state. Use the unchanged frozen `task_normalized_training.train_epoch`: it
already averages pair losses within a task, then task losses within the update.
For T matched tasks, K1's pair weight is 1/T; K2's two pairs each get 1/(2T).
Use identical objective, learning rate, clipping, task order/seed, reference and
one update; both research branches must finish at optimizer step 25. The helper
neither restores nor advances Adam itself. Bind exact original input references
and selected revisions into separate isolated jobs and research ledger claims.
Never write the production checkpoint or latest-optimizer pointer.

Use the original scientific publication/readback mechanisms for both branch
models and Adam exports. Do not replace them with local ledger `complete` flags.
A crash after execution begins needs the same original authenticated outcome;
missing outcome evidence never permits automatic reapplication.

## Evaluation and acceptance evidence

Evaluate both resulting checkpoints on the SAME128 fixed cohort with the same
4db evaluation source, pinned runtime, tasks, prompts, seeds, cap and sampler.
Report per-task paired gains/losses, score uncertainty, gradient norms, reference
margins/drift, cap/EOS rates and actual generation/proof/transfer/training cost.
Do not compare these eight-task training branches against the separate 750-task
cohort as if they share tasks. One research update is a mechanism/coverage
qualification and initial signal, not convergence or a reason to activate K2.

Required next work is the fresh signed research execution binding, original-byte
artifact/readback sinks, frozen-runtime qualification, actual generation and
native/CDF/proof controls, branch optimizer/publication integration and matched
heldout execution. No such GPU/model operations are performed by this patch.
