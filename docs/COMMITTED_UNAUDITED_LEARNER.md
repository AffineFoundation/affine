# Committed unaudited learner

`committed-unaudited-training-v1` is a new signed input contract. Historical
verifier-receipt policies retain their strict audit gate, including E13's original
failure and separately approved startup recovery. This contract must not be
activated by relabeling a historical epoch.

A v2 miner commitment binds each proof ZIP and separate canonical token document.
The token document is limited to 2 MB; it contains the original batch metadata,
not full-vocabulary arrays. The operator authenticates the miner commitment,
freezes small documents, checks schema, token/context limits, permitted mining
indices and held-out exclusions, and excludes duplicated task indices. Its signed
learner admission states **unaudited**. It proves cheap eligibility, not a correct
reward, a target-model execution, or valid sampling. The independent continuous
auditor consumes the same committed proof inventory separately.

The trainer authenticates the exact operator admission and original miner
signature, document SHA/size, and immutable task population before gradients. For
the initial one-turn native MATH contract it also resets the trusted task and
checks the actual task hash, environment seed and exact first prompt against the
pinned tokenizer/template. This requires no generation or outcome grading. Other
environments fail closed pending a trustworthy cheap context adapter.

The existing persistent FP32 master/Adam state and task-normalized preference
objective are unchanged. Claimed success/failure classes remain unaudited and may
contain errors: this is the explicitly selected learning risk, while continuing
audits inform separate reward penalties. Training reports never fabricate
`accepted` audit results or authenticated verifier receipts. Durable checkpoint
and optimizer-state publication and the monotonic parent counter still precede
next-epoch advancement. Empty populations do not count as updates. Independent
evaluation may run separately; audit completion never gates this learner.

The gateway must explicitly provide `capture_learner(epoch)` under the v2 signed
contract. There is no fallback to the large-proof synchronous freeze. Private
`<epoch>-learner-population.json` stores `version`, `manifest`, `submissions` and
`population`; public population history contains original commitments and cheap
eligibility counts, with no proof-validity claim or presigned capability URLs.
The small-input memory reserve covers 256 bounded parsed documents. Runtime/source
hash validation still precedes explicitly declared pure admission bootstrap reload.
