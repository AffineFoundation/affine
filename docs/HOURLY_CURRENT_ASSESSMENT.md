# Hourly current miner assessment

The `hourly-current-assessment-writer-v1` replaces the historical finalized-epoch
payout queue. At each UTC hour it calculates the validator's current estimate
from authenticated eligible task-pair commitments and admitted audit evidence.
Training success, training reports, checkpoint publication and an unfinished
opening are not reward gates. This does not alter historical manifests or claim
that unaudited samples were verified.

Unique eligible task indices are counted at their original commitment time.
Same-epoch task indices contributed by multiple miners score zero. Current audit
evidence supplies the estimated validity and neutral handling of numerical or
infrastructure uncertainty. Contribution counts are multiplied by those current
estimates, then smoothed with a six-hour exponential half-life:

`alpha = 1 - 2**(-1/6) = 0.1091012819`

The moving average uses raw estimated contribution, not previously normalized
weights. Current penalties apply once **after** smoothing, so a new confirmed
invalid result is not diluted by previous rewards. Scores are normalized across
currently registered identities. Audit estimates are cumulative across the
admitted recent history, not reset when a new epoch opens. The signed policy
uses eight recent epochs, audit decay 0.8 and a Beta(1,1) prior. One confirmed
invalid in that recent history multiplies the current score by 0.1; two reduce
it to zero. Three trigger the configured four-epoch blacklist. Infrastructure
errors and numerical ambiguity do not count as confirmed invalids. Historical
evidence retains its original contract and signatures. The bounded seven-day history leaves less than
one billionth of a single old contribution; it is recomputed using current audit
evidence and does not rewrite previous submission records.

The writer wakes every minute to retry chain rate limits, but submits at most
once per UTC hour. On subnet 120 commit-reveal is enabled: the successful
transaction is a timelocked commitment, and the chain later auto-reveals it.
A finalized commitment is not yet an updated visible `Weights` row. The current
chain policy uses one reveal-period epoch and tempo 360; effective rewards
follow that chain schedule. It selects the current hour directly rather than waiting for
an old epoch queue. It holds the existing global writer lock, suppresses the
legacy validator's weight writes, uses an authenticated operator policy and
rechecks hotkey/public-key/UID ownership against the chain. Unregistered
identities are excluded; a race during final rechecking refuses that transaction.
An uncertain transaction outcome requires reconciliation before a retry.

A transport timeout can reuse the last authenticated valid assessment, labeled
stale with its original evidence cutoff. Signature, integrity and computation
errors fail closed. Genuine inactivity decays contribution scores. An explicitly
signed `owner-sink-v1` policy handles a genuine zero-total assessment by assigning
the owner sink, rather than leaving previously penalized miners rewarded. The
default behavior of other ChainAdapter callers remains unchanged.

Each hour retains the signed assessment, source/evidence hashes, estimates,
penalties, exclusions, chain result and last successful submission window.
Deployment uses an immutable local operator runtime and a controlled replacement
of the existing writer service. It does not restart compute or verifier roles.
