# Hourly current miner assessment

The `hourly-current-assessment-writer-never-burn-v3` replaces the historical finalized-epoch
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
identities are excluded. Under the signed `current-hotkey-snapshot-v1` policy,
the adapter fetches one fresh, coherent registration snapshot with signed
activations and both hotkey/UID directions verified at the same block. Departed
hotkeys are removed, retained hotkeys are mapped to their current UIDs and the
remaining points are normalized. A recycled UID never inherits its previous
hotkey's score. Public-identity mismatches and inconsistent snapshots still fail
closed; registration churn alone does not reject everyone else's update.
An uncertain transaction outcome requires reconciliation before a retry.

Weight setting is independent of training qualification. If new evidence cannot
be obtained or authenticated, the writer uses the last authenticated positive
miner assessment, preserves its original evidence cutoff, and reports degraded
freshness. It never uses failed or unauthenticated input as evidence. An empty
new history does not erase valid prior contributions. Current authenticated
penalties and exclusions remain effective when historical contributions are
reused; fallback must not resurrect a miner whose reward was explicitly removed.

Owner-only weights and burn fallbacks are prohibited, including when all scores
are zero. The retired burn command refuses to run, and its service and timer are
masked. The final submission vector excludes the subnet owner. If no positively
scored, currently registered miner remains, the writer reports that condition
and retains the existing chain state rather than inventing recipients. Chain
availability and rate limits can delay submission; such delays are explicit.

Each hour retains the signed assessment, source/evidence hashes, estimates,
penalties, exclusions, chain result and last successful submission window.
Deployment uses an immutable local operator runtime and a controlled replacement
of the existing writer service. It does not restart compute or verifier roles.

The evidence loader also supports exact ROOT-admitted reports from retired
verifiers. This permits only individually pinned original job/report digests;
it does not add retired identities to the active claim roster. Original selected
batch scope, signatures and execution checks remain required. The writer's
signed audit policy is preserved independently of the auditor's policy. This
loader correction passed 25 targeted controls and both same-evidence comparisons.
At the original 23:00 cutoff, 50 report refusals disappeared while all estimates
and weights matched. The combined numerical correction at the prospective 00:00
cutoff changed three estimates and moving averages but preserved weights: newer
unresolved invalid evidence still triggers penalties. The isolated writer package
is installed for 00:00 UTC on 2026-10-07, with the minute timer active. Its first
invocation exited normally while waiting for that hour. A successful new chain
submission under this policy remains to be observed.

The evidence refresh has a 180-second budget, reserving time for transaction
finality. An optional ROOT-authenticated current-hour assessment cache can avoid
recomputing the same history; absence or rejection of that cache does not block
the writer. The cache must match its approved producer, evidence policies,
cutoff, and EMA semantics. Original transaction uncertainty remains fenced until
actual-chain evidence resolves it; old hourly windows are never blindly replayed.
