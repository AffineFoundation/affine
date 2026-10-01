# Tau2 fixed-user search integration

These new modules prepare common-pipeline admission using the original telecom
provider, tools, database and ALL native grader. They preserve the published v1
contracts and earlier controlled evidence.

`native_tau2_common_search_contract` and `native_tau2_common_search_endpoint`
introduce a signed, bounded trajectory attempt. Agent seeds are
`seed_start + task_seed + attempt * 256 + agent_role_ordinal`; auxiliary user seeds
remain `seed_start + task_seed + user_role_ordinal`. The independently approved
user checkpoint, renderer, source, profile and public policy remain fixed when
agent checkpoints or attempts change. Both role receipts and the final audit bind
the attempt. Same-context positive/negative agent decisions can therefore form a
preference pair without placing auxiliary tokens in the loss.

`ops.materialize_native_tau2_common` selects 32 original telecom full tasks after
excluding base tasks: sixteen mining tasks and sixteen disjoint heldouts. Public
records contain IDs and commitments. Raw user instructions, evaluations, original
data and simulation reports stay in operator-owned private files. Materialization
alone does not demonstrate task success or model execution.

`native_tau2_common_simulation` selects the exact committed original task instead
of always taking the first provider task. It retains the original Affine
raw-message/tool-schema patches and runs the native orchestrator in a bounded
trusted child. `native_tau2_common_replay` requires the entire ordered authenticated
request/response sequence, then freshly checks original observations, termination
and grader output. Replay alone does not establish model inference.

`native_tau2_common_driver generate` and `verify` are separate operator commands.
The latter reloads both approved CPU role runtimes, re-renders every complete
request, recomputes full finite float32 output probabilities and strict TOPLOC
proofs, verifies derived responses/public policies, and runs fresh native replay
before signing admission. No truncation or relaxed tolerance is used. Its source
scope is exact declared operator-controlled source, interpreter and package
versions; it does **not** claim a complete transitive binary dependency closure.

The optional first-roaming policy is a disclosed public-input curated computation
control. Agent attempts choose diagnostic guidance; the independently fixed user
algorithm reacts to actual visible tool outputs and instructions. Selected tokens
are verified against the model. This does not prove they were sampled, generalize
the heuristic to all telecom tasks, or establish learning improvement.

These helpers are integration prerequisites. They do not themselves open or
freeze an R2 storage epoch, set proposed miner weights, train, publish a successor
checkpoint, complete a common epoch or submit chain transactions. Those roles
still require the storage/controller bridge and fixed-task evaluation integration.

The first fixed-auxiliary control has now completed genuine generation and a
separate fresh verifier process. It contains six role requests: five user requests
under checkpoint `39818e714a6e4e47b3fdd07e4eeb9cac619cf010fcc83a30068708531eac7d06`
and one agent request under checkpoint
`ad1f39e16f8ec51ad7472d6e7c6e737be0237ef7aae39143ec52b8043c3082d1`.
The agent context is 7,839 tokens plus 31 output tokens; the entire context was
preserved. Independent full probability/TOPLOC recomputation and fresh original
native replay admitted reward 1. This is a controlled positive trajectory, not a
completed common epoch or evidence of learning across telecom tasks.

A separate model-only loader qualification used the saved user ordinal 0 and
agent ordinal 1 computations with those exact checkpoints. It ran in an isolated
package containing the loader and package initializer, without environment
modules or resources. Both full log-probability arrays were byte-equal and both
TOPLOC checks had zero errors. This qualification covers those two computations,
not every trajectory or hardware profile. The existing v2a role-control runtime
has not been silently changed to this new loader.
