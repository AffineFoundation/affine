# Four-sample miner-bound cutover

## Four-sample miner-bound cutover — pending activation

The next contract, `forced-inverse-cdf-prefill-miner-bound-v5`, requires exactly
four distinct rollouts for one task: two native-graded successes and two
native-graded failures (K=2/L=2). Each UID may submit at most three task batches
per epoch. One qualifying unique task earns one contribution point; extra
rollouts do not earn more points. Larger quotas do not establish better learning.

The signed manifest sets `max_attempts=1000`. The existing `rollout.seed` is the
attempt nonce, integer 0–999; its sampler receipt uses `attempt`. It is not an
arbitrary RNG seed. Draws bind epoch randomness, checkpoint, environment/task,
authenticated miner Ed25519 identity, attempt, turn and token position. Use the
authenticated `identity.id`, not a caller-supplied rollout label or mutable UID.
All four rollouts need distinct nonces and distinct generated-token content.
Proofs, filenames, timestamps, nonces and identity prefixes cannot make copied
outputs count twice. The verifier independently recomputes the prescribed draws;
TOPLOC alone is insufficient. Miners may search the allowed attempts and select
qualifying outcomes. This is not unbiased sampling or proof of physical execution.

Activation is pending. Follow the signed OPEN manifest and approved source from
https://affine.io/mining.json. Until it advertises v5, K=2/L=2, max_attempts=1000
and max_batches=3, follow that opening's old rules. Historical signed epochs keep
their original rules and draw bytes. v5 keeps calibrated prefill support with
exact-cached-replay-v1 adjudication; numerical tolerances are unchanged. It does
not activate token-only transport or a new three-way numerical profile.
Training still uses cheap-eligible unaudited inputs independently of continuous
audits. Hourly current-assessment weight setting stays independent of training.

See https://github.com/AffineFoundation/affine/blob/main/docs/FOUR_SAMPLE_MINER_BOUND_CUTOVER.md.

## Boundary and compatibility

Deploy at a new signed opening. Never rewrite an already opened manifest,
previous sampling draws, earned contribution records or optimizer history.
GitHub main alone does not activate v5; bootstrap admits the source of the signed
opening. Old generation/audit paths retain their original context and ceilings.

The miner collects two successes and two failures, deduplicates generated
content independently of sampling metadata, and stops at three qualifying tasks.
An audited batch checks all four outcomes, nonce range and distinctness, output
content distinctness and miner-bound draws. The trainer keeps total task weight
unchanged when expanding a rollout group, and does not reverify sampling.
Grader errors or numerical ambiguity remain indeterminate, not confirmed fraud.

Before activation, authenticate the v5 manifest/source, admit that source on
miner/verifier/trainer routes and observe a real four-sample batch through cheap
eligibility, training and independent audits. Test repacked exact copies, changed
nonces, wrong miner contexts, out-of-range nonces and legacy isolation. Publish
activated status only after the signed opening exists. A four-rollout task batch
is still one batch in the website's batches-per-epoch chart. Keep exactly two
charts: held-out math performance and submitted batches per epoch.
