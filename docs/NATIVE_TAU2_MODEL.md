# Controlled native Tau2 model/proof experiment

This experiment uses the actual original telecom task/orchestrator and ALL grader,
with the approved 135M checkpoint `39818e714a6e4e47b3fdd07e4eeb9cac619cf010fcc83a30068708531eac7d06`
for **both** the target agent and user simulator. This is an explicit controlled
user-model specification, not the original default Engy customer distribution.
It does not modify the running six-environment GPU service or its approved source.

## Genuine inference and complete inputs

`subnet/native_tau2_model.py` serves a localhost OpenAI-compatible endpoint. Each
request commits its complete original HTTP JSON, role, seed and approved checkpoint.
The `native-tau2-complete-chat-tools-v2` renderer passes every original message and
every tool schema into the model's chat template. Structured tool messages preserve
their complete JSON as text; HTTP/sampling metadata remains authenticated in the
request artifact rather than being mistaken for language-model prompt content.

The first canonical whole-HTTP renderer measured 8,205 agent tokens versus the
model's 8,192-token limit. The complete chat/tools renderer measured 7,839 on the
mock transport; actual genuine-model contexts were 3,562 / 7,826 / 3,615 tokens.
No original tool schema, message or condition was truncated. The user has 30 tool
schemas and agent has 13. The generation policy is explicitly versioned eager
FP32 autoregressive KV-cache sampling, temperature0.7/top-p1 with fixed seeds.
Full probabilities and TOPLOC fingerprints are subsequently computed through the
unchanged CPUv2 full-sequence runtime. Independent verification uses that same
strict CPU profile and rejects any TOPLOC mismatch; log-probabilities use atol1e-5,
rtol0. KV-cache sampling does not weaken verification tolerances.

Eight output tokens per request and three bounded native steps produced three
real requests (user/agent/user), 24 generated tokens total. The complete genuine
simulation ended `max_steps`, reward0, with **zero generated tool calls**. This
proves a valid negative trajectory's model/user provenance and original-grader
replay, not a solved task or successful model-generated database mutation. The
separate mock conformance probe exercised an actual original database tool call.
Generation took448 seconds; long strict CPU contexts are expensive. The earlier
uncached whole-request probe timed out without completing an artifact and is not
counted as genuine verification evidence.

## Independent model and native replay checks

`subnet/native_tau2_replay.py` first authenticates the original operator plan,
source closure and verifier supplement, then loads its own approved model and
recomputes all full probability arrays and TOPLOC fingerprints. It next reruns the
original native orchestrator in a fresh subprocess, serving only those verified
responses. Every native request/context must match the committed request exactly.
It compares task identity, complete messages/tool calls, termination and the
original ALL grader's reward information. Only wall-clock message timestamps are
excluded from semantic trajectory comparison.

The verifier independently derives response message content/tool calls from the
captured tokens and action parser. Response role/model/id/usage/finish reason must
match that derivation, and the created timestamp must fall within the signed
request's generation interval. This includes the last user response: a forged
last observation cannot pass just because no later model prompt references it.

## Historical source preservation and stricter audit

The original generation plan/source were preserved. `generation-native_tau2_model.py`
and `generation-native_tau2_replay.py` inside the private evidence folder retain the
actual historical bytes. `operator-source-closure.json` authenticates the original
core/wrapper sources, interpreter hash/Python version and package versions.

The initial verifier omitted exact response/action derivation. Its v2 audit is
preserved as historical evidence and is superseded. The authority-signed
`verifier-supplement.json` explicitly binds the old generation plan/hash to the
stricter `exact-derived-response-v3` verifier/model/replay/attestation source hashes.
It also binds the original closure bytes. This permits a reviewed stricter audit
of an existing genuine artifact without relabeling its generation source.
`native_tau2_attestation.py` enforces this closure before loading model weights;
core runtime/wrapper bytes, interpreter and package versions must still match.

Evidence is under `state/native-tau2-probe/genuine-chat-model/`:

- `plan.json`, `receipts.json`, probability `.npy` files: original signed inputs,
  role outputs and actual probability/fingerprint artifacts.
- `simulation-receipt.json`: original complete native simulation.
- `independent-full-verification-v2.json`: historical weaker audit, superseded.
- `independent-full-verification.json`: latest stricter independent audit.
- `actual-simulation-scope.json`: zero genuine tool calls and negative outcome.

These are operator-private experimental artifacts. No credentials were exported,
no external user API called, no GPU used and no blockchain transaction submitted.
They are not automatically registered as a production native environment.

## Adversarial controls

`ops/probe_native_tau2_mutations.py` requires the completed independent positive
audit, then creates operator-resigned adversarial fixtures so failures test the
computation/semantic checks rather than merely an invalid signature:

- Change the last generated token while preserving probability prefixes: TOPLOC
  must reject the changed activation.
- Forge the last user response and a consistent last simulation observation:
  exact decoded-response binding must reject it.
- Inject a tool call absent from the verified model tokens: the action mapping
  must reject it.
- Forge a simulation user observation: independent native replay must reject it.
- Change reward0 to1: the original grader replay must reject it.

The last two environment-only controls reuse the byte-identical original model
receipts from the completed independent positive model audit, and rerun the real
native orchestrator/grader. No production verification CLI offers a model-check
bypass. The token control performs actual independent model recomputation.
