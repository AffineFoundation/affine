# Prospective full-forward sampling verification

The active E9 contract still uses `forced-inverse-cdf-replay-v1`. This proposal does not change its signed inputs, sampling rules or numerical tolerances.

`subnet/verified_distribution_sampling.py` defines a prospective contract, `verified-full-forward-robust-cdf-v1`. The surrounding verifier authenticates the checkpoint, prompt, environment and complete trajectory, recomputes a causal model forward, and checks probabilities, TOPLOC and environment replay. The sampling check uses those independently computed conditional probabilities, never the miner's claimed arrays. It checks every token against public draws bound to the epoch, checkpoint, environment, task, attempt, turn and position. Token stopping must match the tokenizer EOS or the signed output budget.

This allows a complete causal forward to supply all conditional rows instead of regenerating every prefix. It is a new numerical computation profile, not an assertion that the old prefix replay and full forward are identical. Generation remains autoregressive, and miners must self-check the final complete forward before submitting.

The gate accepts a token only when the draw falls inside its interval for **every** distribution within the pinned absolute log-probability uncertainty. Ambiguous boundaries are unpaid/retry outcomes, not fraud evidence. The existing absolute probability tolerance is 1e-5, relative tolerance zero. The real-arithmetic bound and additional FP64 margin assume qualified CPU `exp`/`fsum`; synthetic controls alone do not qualify those implementations or other GPU families.

## Actual evidence, 2026-10-04

Twenty-five local controls passed, covering authority/context substitution, probability nudging, causal alignment, stopping, bounded inventories, occupied-GPU refusal and original-child supervision.

A private H200 test used the original approved CP093 weights and scientific source, a new nonpayable challenge, native MATH task 6732, temperature 0.8, top-p 1, 128 output tokens and at most eight attempts. The original supervisor observed exit zero without timeout. The first genuine rollout passed all 128 token choices, probability/TOPLOC checks and native grading replay. A synthetic first-token substitution was recomputed through the real model: its probabilities, TOPLOC and grading replay passed, but the new public sampler gate rejected its first choice. Operator readback independently hashed and decoded both complete ZIP artifacts before signing the reference completion.

This establishes a short single-H200 diagnostic, not production qualification, cross-hardware compatibility, a positive/negative training pair, long-context reliability, training convergence or proof of physical historical execution. Original raw artifacts and signed requests are retained privately. B200, L40S and RTX4090 targets must independently pass the same genuine reference and reject the same synthetic control before any prospective hardware admission. A numerical disagreement is a qualification failure, not evidence that an honest miner committed fraud.

A production change still requires a new immutable source, explicit signed computation/sampling profiles, broader genuine and adversarial tests, client compatibility, and a new epoch contract. No current paid batch is relabeled under this proposal.
