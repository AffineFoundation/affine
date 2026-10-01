# Native Tau2 conformance and remaining integration

`ops/probe_native_tau2.py` exercises the original native `tau2.run_task` subprocess,
telecom database/tools, user simulator loop and `EvaluationType.ALL` grader. It is
an isolated transport conformance probe, **not a model-generated training sample**.
Both LLM endpoints deliberately use fabricated responses from a localhost HTTP
server. Reports always label genuine solves, model proofs and authenticated user
observations false. It does not alter the running six-environment GPU source.

## Running the bounded probe

Fetch the wrapper's exact upstream revision into a separate operator-owned directory:

```sh
git init state/native-tau2-probe/upstream
git -C state/native-tau2-probe/upstream fetch --depth 1 https://github.com/sierra-research/tau2-bench.git 337326e62d8e0ca74c353b004a9c5d748e0ba914
git -C state/native-tau2-probe/upstream checkout FETCH_HEAD -- data
printf '%s' 337326e62d8e0ca74c353b004a9c5d748e0ba914 > state/native-tau2-probe/upstream/data/.tau2_revision
.venv/bin/python ops/probe_native_tau2.py \
  --data state/native-tau2-probe/upstream/data \
  --out state/native-tau2-probe/run
```

The data revision marker must match; reports hash every data file and the selected
task, plus installed orchestrator/tool/grader source bytes and package versions.
The task is selected from original telecom `full` minus `base` (2,171 tasks), with
Affine's telecom identifier-example schema patch and upstream raw-message/role
preservation. The original legacy wrapper fixes max steps at 500; this probe
explicitly uses a bounded 8-step budget, max 3 errors and a 90-second process wall
timeout. A timeout kills the whole child process group.

The child imports installed, operator-approved tau2 code. It receives no inherited
API/account credentials; its localhost API key is a constant conformance placeholder.
This subprocess is **not** an execution sandbox for miner-uploaded code. Network
isolation is not claimed. Task data is operator-fetched original upstream material.

## Actual measured result

`state/native-tau2-probe/conformance-final/report.json` records the actual local run.
The first original task is
`[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:None]`.
The probe asks for a phone number and calls original `get_customer_by_phone` with
`555-123-4567`; the actual original database returns customer-not-found. Three
agent and three user HTTP calls terminate with `user_stop`. The original ALL
grader reports reward 0 with failed DB/environment/action checks. This confirms
the real loop/tool/grader path, not task success or an inference proof.

Installed package versions at measurement: tau2 `0.2.1.dev0`, litellm `1.99.0`,
verifiers `0.3.1`. The existing synthetic cache was data revision `798589e...`,
which differs from the legacy wrapper's required `337326e...`; the probe fetched
and used the latter rather than silently mixing cached data. Installed code and
data are separately fingerprinted; a production approved contract must pin both.

## Required before genuine native rollout admission

1. Replace target-agent mock HTTP responses with an approved-model inference
   bridge. Capture the exact serialized tool schemas/messages, token IDs,
   probability arrays and TOPLOC fingerprints for every request. Bind any
   structured tool-call parsing to those captured generated tokens.
2. Pin the separate user simulator model and its settings. Its observation and
   user-tool receipts must be signed by the trusted operator simulator and bound
   to epoch/task/prior-state/request/response, or independently reproducible under
   the approved user model. Target-model proofs alone cannot authenticate user
   observations. Temperature zero is not a determinism guarantee.
3. Independently replay original tool/database transitions and run the original
   ALL grader against the receipt-bound complete simulation; then verify target
   inference proofs. Do not substitute direct agent tool calls for the simulator.
4. Version this native contract separately and stage new signed worker code at an
   epoch boundary. The existing `source-17d...` GPU worker intentionally remains
   unchanged.

No `ENGY`, `OPENAI_API_KEY` or `PRIME_API_KEY` was configured in this probe process.
No API charges, cloud rentals or blockchain transactions occurred.

## Agent and EnterpriseOps inspection

The original Agent wrapper starts its task-scoped MCP tool server as a separate
module; `GeneralAgentToolset.setup_task` dynamically loads task-authored `TaskDB`
and `TaskTools`, and native calls mutate DB state for the original hash grader.
Such code must execute inside a credential-free isolated subprocess/container
with bounded socket/API access, never in the operator process merely because an
uploaded trace names a tool. Its registration bridge must call `_register` so the
native tool catalog is actually advertised.

EnterpriseOps starts digest-pinned service containers and grades final databases
with row-authored SQL verifiers. On this operator host, `docker info` returned
server version `29.1.3`, so absence of Docker is not a measured blocker. Remaining
work is faithful per-task service lifecycle, private seeded databases, merged MCP
catalog, isolated verifier execution and inference/transition receipt binding.
No containers were started by this probe; Agent/EOG production integration is
not claimed complete.

## Follow-up: genuine negative-only model/proof run

The mocked result above is retained as conformance evidence. A subsequent genuine
controlled-user experiment is documented in [NATIVE_TAU2_MODEL.md](NATIVE_TAU2_MODEL.md).
Approved 135M checkpoint `39818e714a6e4e47b3fdd07e4eeb9cac619cf010fcc83a30068708531eac7d06`
generated all three native model requests (user/agent/user), eight tokens each,
using complete original tool schemas/messages and an explicitly versioned
same-approved-model user simulator. The native simulation terminated `max_steps`,
original reward0, with zero model-generated tool calls. It is a genuine negative
provenance experiment, **not K/L positive-batch or native training coverage**.

Fresh strict v3 verification passed all role probability/TOPLOC checks and exact
original native replay/grader checks. The original generation bytes and weaker
v2 verification are preserved; an authority-signed verifier supplement authorizes
stricter response/action binding and enforced interpreter/package/source checks.
Current evidence is `state/native-tau2-probe/genuine-chat-model/independent-full-verification.json`;
historical weaker evidence is `independent-full-verification-v2.json` in the same
folder. The signed original plan, original source closure, historical generation
source copies and `verifier-supplement.json` remain separate immutable artifacts.

Replay the existing authentic artifact using its independently loaded weights:

```sh
MKL_CBWR=COMPATIBLE ATEN_CPU_CAPABILITY=default ONEDNN_MAX_CPU_ISA=SSE41 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
.venv/bin/python -m subnet.native_tau2_replay \
  --out state/native-tau2-probe/genuine-chat-model \
  --checkpoint state/service-conformance/checkpoints/nonpayable-service-conformance-1790827619-9 \
  --data state/native-tau2-probe/upstream/data \
  --authority d54a3a345d0de3e2c7898f30c0942d78f931f8c4b8036ffdc6adffcd2525062f
```

The generation/source artifacts stay historical. A source change requires a new
reviewed signed verifier supplement; changing their labels does not authorize
new code. This experiment has not been added to the running GPU source or to
production native-miner scoring.
