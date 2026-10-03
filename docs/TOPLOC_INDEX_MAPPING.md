# TOPLOC 0.1.6 index-map replay

Large prefill activation chunks can contain selected positions that collide
modulo 65497. TOPLOC's `ProofPoly.from_points` chooses an injective index-map
modulus and stores it in the proof header. Polynomial coefficients are still
computed in the fixed 65497 field. Its native C++ verifier applies the stored
index map before polynomial evaluation; its Python list verifier does not.
That omission can reject a proof against the identical tensor used to build it.

`subnet.proofs.verify_mapped_proofs` follows the native C++ semantics: flatten
each original prefill/decode chunk, select the same top-k positions, map those
positions by the encoded modulus, then evaluate the original coefficients.
It rejects noninjective maps and malformed framing, keeps explicit native
bit-extraction thread limits, and returns the original error statistics.
Runtime verification still requires zero exponent mismatches and zero mean
and median mantissa error. Full-vocabulary log-probability checks, checkpoint
hashes, environment replay, model inputs and sampling are unchanged.

The actual original-MATH task 6665 diagnostic reproduced identical generated
and replayed hidden bytes and zero probability error, while the old verifier
rejected only prefill block zero, even against the original hidden tensor.
The original failed evaluation is preserved as an error, not relabeled as a
complete baseline. A new signed source, fresh worker admission and actual
GPU qualification are required before deploying this correction.

Regression tests construct a real BF16 index collision, demonstrate the old
Python rejection, compare the corrected result to TOPLOC's native verifier,
and reject changed activation values, altered polynomial coefficients and
invalid proof headers. Normal prefill and partial decode blocks retain their
previous exact results. These tests establish the codec correction; they do
not establish cross-hardware inference equivalence or model improvement.

On October 3, 2026, an independent retained H200 verifier node ran the
corrected source `24d3638ff736b21b5ca1232b0da2f8fa980a266a8b2059ce85d61aef7bac6533`
against the untouched Qwen2.5-Math-7B checkpoint. Task 6665 with seed 26926002
reproduced the original diagnostic's hidden-byte digest, 87 prompt tokens and
218 generated tokens. The prefill proof selected modulus 65496. All 15 proof
blocks passed with zero errors, and full-vocabulary probability error was zero.
Changed tokens, changed log probabilities, changed proof bytes and changed
hidden activations were each rejected. Root independently checked the signed
job, pinned source and package versions, helper digest and result bindings.
No optimizer ran and no thresholds changed. This qualifies that concrete
pinned H200 computation; fresh paired evaluations remain necessary for a
performance comparison.
