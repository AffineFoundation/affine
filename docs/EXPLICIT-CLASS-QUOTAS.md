# Prospective class quotas

GPU operator configuration accepts explicit integer `K` (positive rollouts) and
`L` (negative rollouts). Each defaults to 1. The effective sum must be at most
128 and fit `sampling_policy.max_attempts` when prescribed sampling is enabled.
Malformed, boolean, null, zero and infeasible quota values reject before epoch
upload capabilities are created. Explicit values pass through `contract`, the
initial manifest and `Controller.open` into the original signed public manifest.
Unconfigured defaults retain the previous contract kwargs and manifest bytes.
Historical signed manifests and their quotas are not rewritten.

The existing training adapter zips the positive and negative lists in their
original order. It consumes `min(K,L)` pairs per eligible batch. Thus K2/L2
consumes two pairs, while K2/L1 still consumes one; the extra positive is retained
in the original claimed batch but is not a second training pair. Existing
`pair_count` diagnostics report consumed pairs, independently of submitted
batch/document counts. Task-normalized training keeps each task's aggregate
weight unchanged when its number of pairs grows. This change does not alter the
trainer, scoring, quotas on active epochs, or its reference log probabilities.

This plumbing is prospective. A first ordinary search-budget trial can change
only owned `search_budget` from 8 to 16 while retaining K1/L1 and the already
signed maximum of 16 attempts. A K2/L2 trial requires a new explicit opening;
setting K/L on an operator config no longer silently produces a K1/L1 manifest.
No source qualification, GPU measurement, live config change or activation is
claimed by these CPU controls.
