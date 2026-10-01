# Prospective common Tau2 role contract

This contract prerequisite is implemented only in the new
`subnet/native_tau2_common_contract.py`. It does not change the frozen native
Tau2 model/replay/auxiliary-role adapters, launch a broker, run CUDA, admit a
production epoch, or train a model.

The historical controlled native Tau2 evidence verified agent and user model
computations against the same approved checkpoint, replayed the original
orchestrator/tools/grader, and separately qualified masked agent training.
Those results remain historical controls. They cannot be relabeled as the new
fixed-auxiliary common epoch contract: its signed receipt and audit fields must
be generated and independently verified by a future two-runtime bridge.

## Roles and immutable auxiliary policy

The signed manifest version is
`native-tau2-common-fixed-auxiliary-contract-v1`. It contains an epoch ID,
current agent checkpoint descriptor, environment version/source closure,
taskset and original data inventory hashes, and exact task indices, task hashes,
and seeds. Its objective is `agent-only-native-outcome-preference-v1` and its
controlled scope is nonpayable with no blockchain transactions.

Exactly two model roles are declared: `agent` is training eligible and `user`
is auxiliary and ineligible. Each role binds exact checkpoint files and their
canonical content ID, runtime profile and package/interpreter pins, source
closure, complete-message/tool renderer, harness source hash, vocabulary and
context/output limits, strict numerical policy, and seed policy. The numerical
policy retains log-probability atol 1e-5, rtol zero, and zero TOPLOC errors.

The operator separately supplies `approved_fixed_user` to validation. This
immutable trust anchor includes the auxiliary checkpoint, renderer, harness,
source, runtime, and seed policy. It is never accepted from a miner sidecar.
The manifest's user descriptor must match it with canonical JSON type equality.
Changing the current agent checkpoint does not change this auxiliary model.
Changing the auxiliary policy requires a separately approved contract/evaluation
group, rather than silently modifying a held-out baseline. The pinned user
model can still respond differently to different agent actions: its conditional
inputs remain the complete authentic native context.

Seeds use `task-seed-plus-role-ordinal-v1`: role seed start plus signed task seed
plus that role's request ordinal. This keeps user randomness independent of the
number of preceding agent requests. Ordered global and per-role ordinals are
both checked. No message, tool schema, output token, or observation may be
removed to fit a context budget; an over-budget sample is rejected.

## Signed audit sidecar

The sidecar version is `native-tau2-common-fixed-auxiliary-audit-v1`. It binds:

- Canonical manifest hash and hash of the ordered **signed** role receipts.
- Exact raw verification report hash, epoch, environment/version, task hash,
  environment index, original reward, and explicit curated-computation scope.
- Complete native trajectory, all model roles, derived response mappings, and
  approved source closure verification, each as an operator-attested result.

Each signed role receipt binds the epoch/task, role descriptor, exact checkpoint,
profile/source/harness/renderer, full request/response hashes, exact input/output
tokens and hashes, seed and ordinals, full probability file hash, and proof
artifacts. The independent report has one ordered `role_checks` row per signed
receipt, bound by signed-receipt hash, with model/context/derived-response checks.
Every receipt requires complete coverage; a context-blocked or missing role is
not admitted as a full trajectory. Reward remains tied to original native replay.

`admit_sample` authenticates this lineage against the separately trusted audit
signer. It does **not** recompute TOPLOC, read the probability arrays, derive an
OpenAI response from decoded tokens, or replay native tools itself. The future
operator verifier must perform those operations before issuing its signed audit,
including the strict last-response/tool-call binding from qualified Tau2 v3.
A signed statement alone is not a cryptographic execution proof.

## Training views and integration gate

Admission derives a loss mask for every output token. Agent tokens are true;
all auxiliary tokens are false. Any submitted conflicting mask is rejected.
The auxiliary evidence remains in the artifact for observation authentication.
`preference_pair` accepts the admission results within the trusted process and
selects a divergent agent decision with exactly the same full token prompt,
same task/epoch/current agent checkpoint, and same immutable user policy.
Externally serialized training views must be re-admitted from their original
signed receipts/audit before use; a plain descriptor is not an authority token.

This is curated outcome-conditioned preference data, not a claim of unbiased
or originally sampled trajectories. Ten synthetic signature/lineage controls
cover fixed-user drift, agent checkpoint updates, malicious loss masks, signer
substitution, changed reports, incomplete role checks, source/context/seed/model
mismatches, strict types and budgets, and same-prompt agent-only preference.
They run without a model or native service and do not establish common pipeline
coverage. Next gates are a trusted two-model endpoint/broker, independent role
proof and native replay reports under this exact schema, and a real signed
common epoch that consumes the admitted agent-only views.

## Separate bounded-search contract and endpoint (v2)

The new `native_tau2_common_search_contract.py` and
`native_tau2_common_search_endpoint.py` preserve v1 and introduce a signed
trajectory attempt. The agent seed adds `attempt * 256`; auxiliary seeds retain
the original task-plus-role-ordinal rule. The signed maximum attempt count is
bounded at 1,024. Receipts, reports and audits bind the exact attempt, so a
successful trajectory cannot be substituted for another search attempt.

The endpoint accepts two operator-owned approved runtimes. Before every model
call it checks the role descriptor and declared source bytes. It renders the
complete native request, records every emitted token, stores full finite float32
log probabilities and TOPLOC fingerprints, and binds the derived text/tool-call
response. Independent `verify_receipt` re-renders the context and recomputes both
probabilities and fingerprints under the pinned tolerance; complete native
trajectory replay and aggregate signed admission remain separate requirements.
Neither one valid receipt nor a signed manifest establishes a complete rollout.

User output loss masks are always false. Preference selection operates only on
already admitted views within a trusted process and requires a divergent agent
decision under an identical token prompt. Serialized views must be readmitted
from the original signed receipts and independently verified audit; ordinary
Python dictionaries are not proof objects. Source validators, renderers, framing
validators and candidate-policy hooks are trusted injected operator dependencies,
not values accepted from miner artifacts.

Thirty-three synthetic contract/endpoint controls pass, covering lineage,
independent user policy, attempt seeds, complete context, response mappings,
probability/proof mutations and auxiliary loss masks. These controls do not
constitute native model execution, a storage epoch, training or improved held-out
performance. The actual native integration is being qualified separately.

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_native_tau2_common_search*.py'
```

## Original task inventory and native replay helpers

`ops/materialize_native_tau2_common.py` materializes original pinned telecom
full tasks while excluding base tasks. By default it selects 32 tasks: sixteen
mining indices and sixteen disjoint held-outs. The public file contains only task
IDs and hashes; raw instructions, criteria and data remain operator-private.
Re-materialization rejects conflicting existing task commitments.

`native_tau2_common_simulation.py` verifies the signed task/data/source bindings
and supplies the exact selected original task to the original orchestrator with
its original Affine message/tool patches and ALL grader. A bounded trusted child
uses only the loopback role endpoint. It is not a sandbox for uploaded miner code.
`native_tau2_common_replay.py` authenticates the full ordered role sequence and
feeds its responses into a fresh original run. It checks every native request,
message/tool observation, termination and reward field; only wall-clock message
timestamps may differ. This replay cannot attest model computation and explicitly
leaves those admission flags false until a separate numerical audit completes.

Twelve synthetic controls cover selected-task routing, private/public separation,
provider drift, conflicting task commitments, exact replay requests, observations
and grader output. Genuine execution evidence remains distinct from these tests.
