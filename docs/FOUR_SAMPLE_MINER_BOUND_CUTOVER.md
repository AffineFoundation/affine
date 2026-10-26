# Four-sample miner-bound cutover

## Manifest-driven quota support (not yet activated)

Set only `samples_per_batch` in prospective controller configuration, for
example `"samples_per_batch": 8`. Omit separately configured K/L; the publisher
derives K=L=4 and includes both counts and the total in the signed manifest.
Changing that value to 16 derives K=L=8 without code changes. Valid totals are
even integers from 4 through 128; contradictory explicit counts are rejected.
The native multi-rollout v3 policy can use `"max_pairs": "manifest"` to derive
its bounded pair budget from the same signed quotas and the 256-task training
ceiling. Existing explicit native budgets and historical manifests are preserved.
This is configuration support, not a claim that larger groups fit every machine
or improve learning. Changes take effect only in a newly approved signed opening.

Updated software supports larger balanced v5 quotas, including K=4/L=4:
eight distinct rollouts per task batch, four successes and four failures.
The limit remains three task batches per UID per epoch, with attempt nonces
0–999 and one contribution point per qualifying unique task. Miners read K/L
from the signed manifest; eight samples are still one batch, not eight points.
Existing signed openings and source archives remain unchanged. External miners
must use the newly approved source when a future opening activates larger quotas;
old clients with fixed four-sample checks cannot adopt it by manifest alone.
The current deployed opening remains K=2/L=2 until a qualified boundary cutover.

## Four-sample miner-bound cutover — activated in epoch 53

The epoch-53 contract, `forced-inverse-cdf-prefill-miner-bound-v5`, requires exactly
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

Activated in signed epoch 53 on 2026-10-07 at 20:19 UTC; the public discovery
endpoint independently advertised the same opening. Its approved source SHA256 is
`0dcf31a608fe86075b6b8e755486978d62ebf8a2d8518e166ce4fc83e942f360`.
Follow the latest signed OPEN manifest and approved source from
https://affine.io/mining.json; it determines current availability and deadlines.
Epoch 52 was closed after a pre-GPU bootstrap failure and was not relabeled.
Historical signed epochs retain their original rules and draw bytes. v5 keeps
calibrated prefill support with
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
