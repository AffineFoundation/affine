# Fixed-reference epoch optimizer qualification

`subnet/epoch_optimizer.py` defines the versioned policy
`bf16-full-adamw-fixed-epoch-reference-v2`. It computes every verified pair's
reference margin from the immutable epoch input before any update, then keeps
one AdamW instance across the requested steps. Sequence scores are mean output
token log probabilities; environment messages contribute context, not loss.
Every parameter must receive a gradient, and every Adam state must advance on
each step. Existing deployed source forks are unaffected by this module.

The retained RTX 3090 qualification used the independently verified original
Spider SQL K1/L1 artifact and a 134,515,008-parameter approved model. Three
actual full-model updates produced losses 0.693147, 0.692800 and 0.692718. All
272 parameter tensors received gradients each time, optimizer counters advanced
through 1, 2 and 3, and the reference remained fixed. Peak allocated memory was
1,438,733,824 bytes. A fresh model process verified an updated-model rollout's
full probability arrays, TOPLOC and original environment replay.

The resulting checkpoint is
`3313407f426fd3393d861f219003bbb57a0395719448c61e72e135b73ad53853`.
Independent operator inspection authenticates the signed job, frozen source,
actual audited input artifact, update attribution, reference calculations and
reported output file map. These are controlled operator-collected execution
records, not a cryptographic proof of GPU execution. Independent streaming of
this successor's actual weight bytes remains a separate publication gate.

This establishes training-pair progress on the small model. It does not
establish held-out improvement, a completed common epoch under this new policy,
fit on the wider 1.7B model, or a new hardware profile. Prospective controllers
must select a newly pinned policy/source only at a completed epoch boundary.
The qualification makes no blockchain transactions.
