# Prospective v4 verification and v3 resolution coverage

These changes are default off and do not activate a source, sampler or payout.
The explicit `forced-inverse-cdf-prefill-threeway-v4` contract continues to require
compact selected-token probabilities, TOPLOC, pinned checkpoint/calibration,
prescribed public draws and native replay. No cached generation fallback is added.

A v4 CDF numerical ambiguity is saved while the verifier checks all remaining
turns and native outcomes. The batch auditor likewise checks every remaining
rollout and the positive/negative quota. A confirmed failure takes precedence
over an earlier unknown. When all other checks pass, uncertainty remains
`numerical_ambiguous`, not accepted or fraudulent. Signed unknown reports include
per-rollout/per-turn uncertain positions and whether native replay completed.
The v2/v3 sampler contracts keep their previous immediate uncertainty behavior.

The separately selected `continuous-probabilistic-audit-v3` preserves the existing
validity posterior and cheating penalties, then multiplies raw points by the
smaller of current and decayed recent resolution coverage:

`resolved / (resolved + numerical_ambiguous)`

Resolved includes authenticated valid and confirmed-invalid observations;
confirmed-invalid penalties still apply separately. No Beta prior counts as a
resolved observation. An unknown-only current cohort earns zero raw points even
with a strong valid history. Infrastructure errors and unselected/pending rows do
not count in either denominator, receive no fraud penalty and cannot resolve an
unknown. With no scientific observations in a window its coverage defaults to one,
preserving prior-based credit for unaudited supply. Recent unknowns can discount a
new epoch until resolved under the existing explicit adjudication contract or aged
out. All observations retain existing cutoff, deduplication and immutable evidence
bindings. Ordered cohort hashing is the same as v2. Historical v1/v2 snapshot bytes
and draw domains remain unchanged.

This coverage discount measures resolution, not a cheating probability. It may
favor numerically stable tasks, and an all-unknown network yields zero weights;
ROOT should review this incentive choice and honest uncertainty rates before
selecting it in a prospective signed policy. Existing chain activation and immutable
opening/source approvals still apply. No research GPU result is relabeled as
qualification of these changed verifier/report modules.
