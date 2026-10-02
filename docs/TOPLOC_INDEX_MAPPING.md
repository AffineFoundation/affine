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
