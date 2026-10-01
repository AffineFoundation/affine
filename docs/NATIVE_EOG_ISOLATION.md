# Original EnterpriseOps Calendar controls

The controlled Calendar harness runs the **original** digest-pinned
EnterpriseOps-Gym Calendar API and MCP tools. Five genuine original Calendar
tasks were materialized from dataset revision
`c8e538eae8a6205294f0a86675fefdc1fac408f6`; their SQL seeds match the exact
archive at repository revision `de22905d21a080b83bf4a54258afe4250ee2dd55`.
This experiment does not change the shared environment registry or active
workers, and has not generated model proofs or trained a checkpoint.

## Actual controls

Four original tasks reset in independent, newly seeded services and produce
their original reward zero. A fifth task (Calendar index 54, relocate office
events) executes six real tools using the original public instruction and
actual tool results: calendar discovery, event discovery, and event patches.
Its original SQL verifiers change from reward **0 to 1**. Replacing the requested
destination with a wrong location produces reward **0**. Neither policy reads
private SQL verifier definitions or raw seed state.

Fresh replay of the successful six-tool trace reproduces every observation
byte for byte, the complete logical database state, and original reward 1.
Fresh services reject five falsifications: forged observation, changed tool
arguments, false reward, false logical state hash, and changed clock profile.
The six-tool move/title/ACL task also has exact replay and five rejected
falsifications, while retaining its genuine original reward zero.

The latter task exposes an unresolved native semantic discrepancy: its ACL
tools return success, while its original SQL checks do not observe the requested
matching ACL rows. The calendar-creation task also has an original exact-string
datetime check that does not match the native SQLAlchemy timestamp formatting.
These outcomes are preserved in private evidence; API and grader code were not
changed to manufacture passing results. They do not prevent the relocation
task's genuine positive and negative controls.

## Runtime and isolation

The base image is
`shivakrishnareddyma225/enterpriseops-gym-mcp-calendar@sha256:994c5421a6dd065861bc7f813a177f6d408875e9df60fe8d012959bc4510da02`.
All **93 original bytecode files** have identical hashes in the derived image.
The separate `native_eog_clock.py` entry point defines a disclosed
`eog-calendar-fixed-clock-uuid-v1` profile: approved fixed UTC wall-clock,
seeded UUID stream, and fixed SQLite `datetime('now')`/`CURRENT_TIMESTAMP`.
Other SQLite date operations delegate to the original SQLite engine; real
monotonic clocks and timeouts remain unchanged. This profile must be pinned in
an operator-approved environment descriptor, never selected by the miner.

Services run as the image's `calendar` user with network **none**, read-only
root, no host mounts or published ports, all capabilities dropped, and bounded
CPU/memory/pids/tmpfs. HTTP requests execute **inside** the container. The actor
interface accepts only the original selected, unrestricted tool names and JSON
arguments. It cannot request SQL, seed/reset operations, arbitrary HTTP routes,
or shell commands. The private original `run_verifier` and Affine `solved`
function bodies execute in the operator process; only transport is replaced
with inside-container HTTP. Private queries and expected values never enter
model context or the derived image.

Logical state identity includes `service`, `database_id`, `table_counts`, and
every `table_data` row. The API's operator-only filesystem metadata (including
database file mtime) is recorded separately and excluded from logical identity.
No tool observation, action, state row, or original verifier result is normalized.

## Reproduce

From the repo, using the existing pinned dataset/archive cache:

```bash
.venv/bin/python -m ops.materialize_native_eog
.venv/bin/python -m ops.build_native_eog_runtime
.venv/bin/python -m ops.probe_native_eog
.venv/bin/python -m ops.probe_native_eog_replay
.venv/bin/python -m ops.probe_native_eog --relocation-controls
.venv/bin/python -m ops.probe_native_eog_replay --relocation
.venv/bin/python -m unittest discover -s tests -p test_native_eog_isolation.py
```

The successful relocation policy is `public_relocation_policy` in
`ops.probe_native_eog`; the relocation commands materialize its positive/negative
artifacts and verify the positive trace and falsifications. Raw fixture, database, and grader evidence
stays under `state/native-eog-isolation`, outside source archives/public feeds.
It is operator-collected evidence, not a cryptographic proof of execution.

## Common pipeline admission still required

Pin original source/data hashes, derived immutable image, shim hash, seed and
clock in the signed environment specification. Keep the private verifier
catalog separate from the public actor bundle. Add a prospective adapter in an
isolated source bundle that serializes the exact public messages/tool schemas,
uses original MCP observations, and delegates replay to this fresh native
runtime. Capture actual target-model tokens, full probabilities and TOPLOC
fingerprints; independently verify those against each exact tool context.
Only then demonstrate shared epoch upload/audit/proposed weights/training and
held-out comparisons. Current controls cover Calendar only: the other original
service families, two-service hybrids, model proofs and common epochs remain
unverified. No chain transaction or original production service was touched.
