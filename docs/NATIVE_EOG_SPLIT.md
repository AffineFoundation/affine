# EOG public actor / private original grader

The original version-2 source is preserved at Git commit `3a0cfc09` and in
`state/native-eog-split-v2-frozen/source`; its controls and independently verified
native replay remain historical evidence. The current prospective
`original-eog-public-actor-private-grader-v3-terminal` profile adds a terminal
scalar outcome and writes separate reports under `state/native-eog-split-terminal-v3`.

This prospective boundary leaves the frozen Calendar version-1 controls and
active shared workers unchanged. It separates the miner-facing actor RPC from
operator-only fixture seeding and original SQL grading. The tested scope is an
operator-owned, co-located pilot; an untrusted host administrator could inspect
its processes. External miners require the broker on an independently trusted
operator host. No host-root isolation claim is made.

## Contract

`OperatorBroker(private_task, runtime)` is created by the trusted operator.
The private fixture, SQL seed file, context headers, verifier queries and
expected values remain in this owner. The original Calendar service remains
network-none, nonroot, read-only, without host mounts or published ports. HTTP
calls to the native service execute inside its container.

`PublicActor(endpoint, actor_capability, expected_descriptor_sha256)` stores
only those three values. Its versioned descriptor contains exactly schema,
revision, task ID, original public messages, all selected unrestricted original
tool schemas, approved runtime, seed **hash**, and source-file hashes. It has no
raw seed, seed path, SQL verifier, database snapshot or operator capability.

The `/actor` RPC accepts only:

- `reset`: returns the approved public descriptor before any action.
- `call`: an original selected tool name and JSON arguments.
- `close`: seals further actor writes; it does not expose a grade.
- `finish` (version 3): seals writes and returns only the original scalar reward,
  opaque session/task ID, public descriptor hash, and exact transcript hash.
  Repeating finish returns the identical terminal response; raw verifier
  results remain private. No intermediate reward-oracle operation exists.

Selected native calls retain their original observation bytes. Unknown or
unselected names return the exact pinned FastMCP `Unknown tool: NAME` error,
without forwarding SQL-shaped requests to the service. Actor reset after
mutation, writes after sealing, extra operation fields, raw routes, state reads,
seed/reset requests and file reads are denied. Sessions have a 32-call budget.

The separate `/operator` capability permits original grade and logical state
hash retrieval. Its response contains private original verifier results; this
is never returned to the actor. Capabilities are independent 256-bit random
values, are not put in model messages or public reports, and authorize only the
single broker session. The controller must call `OperatorBroker.close()` to
remove its own service after grading; actor close deliberately preserves it for
operator assessment. The pilot RPC listens only on host loopback.

## Proven native behavior

The genuine original Calendar relocation task (index 54) runs six native tools
through the actor-only client. Original reward is **1** for the requested
destination and **0** for a wrong destination. A fresh broker reproduces the
complete positive observation trace, original reward and logical database hash.

The actual controls deny ten privileged/unselected/sealed requests and reject
three fresh native replay falsifications (observation, reward and logical state).
They verify original Calendar execution, not target-model generation. Five unit
tests additionally cover descriptor leakage and authenticated model-audit byte,
weight, trajectory, authority and numerical-policy bindings; their synthetic
receipts are explicitly unit controls rather than real inference evidence.

Run `.venv/bin/python -m ops.probe_native_eog_split` and
`.venv/bin/python -m unittest discover -s tests -p test_native_eog_split.py`.
Version-3 reports are in `state/native-eog-split-terminal-v3`. Operator-private reports are
mode 0600. No capabilities are written to reports.

## Coupling independent model verification

The sanitized original six-tool input is
`state/native-eog-isolation/relocation-model-input.public.json`. Its SHA256 is
`c71cdacc7537f8a0192bcd501c746a7a5bfdc89725dcd18dfbdf810c26cb7bea`.
It contains public messages, complete schemas, actions, actual observations,
claimed reward and runtime/source/seed hashes, without grader or seed content.
Do not send the similarly named `.private.json` artifact to a model worker.

`native_eog_admission.validate_model_audit` verifies a signed receipt against an
independently approved authority and model/profile/weight inventory. It binds
all six complete message contexts, actions and raw observations, requiring
strict full-probability/TOPLOC verification and an explicit curated-computation
scope rather than original-sampling provenance. `admit` then independently
replays those six tools using the split broker and checks the original reward.
It emits source, runtime, image, seed, task, native state and signed model-audit
hash bindings. The model receipt is authenticated evidence; admission does not
itself recompute activations or constitute a cryptographic GPU execution proof.

The old 8,192-token model profile cannot cover the full trajectory: its complete
native contexts reach 16,973 tokens before the final output. A separately pinned
larger-context model/profile is required. Neither source nor context may be
silently truncated. Subsequent controlled Qwen model audits now cover both
six-tool positive and negative traces under the exact frozen version-2 source.
Fresh native admission reproduced their original rewards 1 and 0. These are
curated target-model computation controls with complete contexts and probability
records, rather than original sampling or version-3 terminal-output evidence.
Common epoch uploads/audits, proposed weights, training and held-out comparisons
remain separate gates.
