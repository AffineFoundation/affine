# Original Spider grader isolation

`native_sql_isolation` runs the original `SqlTask.correct`, SQL extraction,
scratch-database query execution, normalization and result-comparison bodies
inside a separate digest-pinned Python container. Their original bodies are
compiled directly from the vendored source. Only the framework reward decorator
is omitted; the standalone caller provides its task/trace attributes.

The private grader receives the original database bytes, reference query and
submitted final reply. The reference query is never a miner prompt. Every run
has a fresh database and scratch files, no network, no host mounts, a read-only
root, an unprivileged user, no capabilities, and bounded memory, processes,
CPU and temporary disk. A separate wall-clock deadline removes the exact owned
container even when a recursive query does not terminate. Resource failures
are rejected execution failures, rather than manufactured native reward zero.

Run `.venv/bin/python -m ops.probe_native_sql_grader` using the existing cached
original Spider train dataset and database archive. The control uses the first
original training task: its reference query is a positive **grader control**,
not a generated model sample. An incorrect query and invalid file/extension
function calls retain the original reward zero. A nonterminating recursive query
tests the execution deadline. Fresh positive evaluation afterward confirms the
original database was unchanged. Private fixtures and reports stay in `state/`.

This is a grading prerequisite. It has no inference proofs, generated positive/
negative model pair, common-epoch training or full original bash-harness
admission. The next integration must provide an isolated public actor containing
the original database and tools, preserve native prompts and task identities,
and independently audit actual model computation and tool observations before
any batch is scored or trained.
