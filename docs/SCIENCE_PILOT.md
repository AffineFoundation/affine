# Original Science admission pilot

The prospective Science taskset contains thirty-two original INTELLECT-3-RL
science problems from the prior Affine provider. Mining indices are 0–15;
held-out indices are 16–31. The snapshot preserves original questions, system
instructions and answers. All thirty-two tasks reset successfully, have distinct
task hashes, and returned zero through the original grader for a deliberately
invalid unboxed answer. Maximum public prompt length is 1,146 characters.

These are environment controls, not model samples, positive/negative batch
admission, training or learning evidence. Private taskset and control evidence
are under `state/science-tasksets`. Original source hash is
`5942dfa615e8bbedf4aec05c8a742b784a45557934317cc3065a44f09db6de67`;
snapshot hash is
`479cd1a61b49e9682078d6487f9787812c366cdc11bf956aa7b381786b1f5fbf`.

`ops.probe_text_model_search` now accepts this source and Trivia Abstain,
alongside the already tested Trivia and PopQA sources. An operator-signed
plan pins source membership, weights, original task snapshot, harness and
numerical profile. The admission probe requires autoregressive sampling with
no per-turn overrides, training-only indices and a bounded search budget.
A separate process reloads the model for full probability/TOPLOC verification
and original environment replay. Common-epoch admission requires a genuine
positive/negative pair; no such pair is claimed by these environment controls.

The original Science wrapper uses boxed-answer math verification without an
LLM judge fallback. Its reward can undercount valid unit or expression answers;
the pilot retains that original behavior rather than changing the grader.

The bounded remote model probe has now completed on the approved 1.7B
checkpoint `89995d56ebb787252f9ed71eace4d7ae7e8c4ff975ae634c4752beec03f2247d`.
It sampled eight autoregressive seeds for each of original mining indices
0 and 1, with a 256-token output budget. All sixteen attempts were negative.
One negative rollout per task was retained; a separate model reload verified
both full probability arrays, strict TOPLOC fingerprints and original replay.

Root independently authenticated the signed plan and collected receipt,
checked all 42 pinned source modules and both actual ZIP artifacts, and
checked every retained float32 probability array and proof frame. Root also
replayed both retained outputs through the original local Science grader;
rewards, classifications and observations matched exactly. That local control
has a distinct adapter/dependency source hash from the remote frozen source;
both preserve the same original task snapshot and task identities. This
inspection did not recompute the GPU model locally.

This probe provides genuine negative-sample verification evidence, but no
positive/negative pair, common training epoch or performance improvement.
Evidence is in `state/science-tasksets/model-control/root-artifact-evidence.json`
and `root-fresh-native-replay.json`. Mining admission remains unproven.
