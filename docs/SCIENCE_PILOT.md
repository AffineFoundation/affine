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
