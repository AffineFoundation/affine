# Reviewed historical TOPLOC uncertainty

This candidate is default off. No production service, original report, or past
submitted assessment changes when this code is installed. Activation requires
a new signed ROOT writer policy, pinned execution modules, a signed numerical
resolution policy, and authenticated byte-exact reference archives.

The four independent H100 replays reproduced the original TOPLOC rejection.
Their mismatches were small: mantissa mean at most 1/128, median zero, and at
most one exponent mismatch per segment. This supports uncertainty about the
origin of the deviation. It establishes neither malicious conduct nor a valid
rollout. In particular, `Runtime.verify` raises at TOPLOC before later sampler
and grader checks finish. The overlay explicitly marks those later checks as
**not established**.

`reviewed-toploc-numerical-resolution-v1` is a separate signed policy. Its
entries bind the exact epoch, checkpoint, frozen source, commitment evidence ID,
original job, original observation digest, report and report-request digests,
artifact, independently reviewed result, and full-readback archive ACK. Every
entry has an integer review time at or before the new assessment cutoff.
Supported entries change the effective category only from the original fully
audited `InvalidSample: TOPLOC` to `numerical_ambiguous`. Originals remain in
the admission record and the output identifies the original category and
review digest. Entries cannot manufacture `verified_valid`, resolve structural
failures, resolve another native error, or use infrastructure errors as fraud.
The existing ambiguity-only `continuous-audit-adjudication-v1` contract remains
unchanged.

The reference archive is checked again: ROOT ACK signature and full SHA,
ROOT reference scope, original zero-exit terminal, exact result and original
bindings, complete native segment metrics, and the narrow uncertainty limits.
Unsigned or substituted policies, changed tuples, late reviews, absent module
pins, unsupported outcomes, and larger fingerprint differences are refused.
Prior-cohort snapshots can omit later populations; an unknown evidence ID in a
present epoch is still refused.

Under the required continuous audit v3 policy, these observations contribute
neither success counts nor confirmed-invalid counts. They do affect unresolved
coverage. An all-unknown miner therefore receives no fabricated valid coverage
and no fraud-based blacklist. Current assessments reuse this same effective
evidence. Past on-chain transactions and immutable assessments are untouched.
The writer's old v1 exact schema remains supported; the new explicit version is
`hourly-current-assessment-writer-reviewed-numerical-v2` with the additional
`numerical_resolution_policy_sha256` field. Its execution module pins must
include the actual writer, evidence reader, audit policy, and overlay module.
The signed service configuration supplies `numerical_resolution` with
`policy_document` and bounded `reference_archives` entries containing
`ack_path` and `archive_path`. Without the independently signed writer policy
pin, that configuration is ignored.

Nine original epoch-29 TOPLOC rows were inventoried and authenticated. Four
have independent reference evidence; the other five cannot inherit those
results and require their own bounded, ROOT-authorized research execution.

Before a future production verifier contract changes, it should report native
mismatch magnitudes directly. A narrow numerical uncertainty should be carried
through the sampler and original grader checks, so a later clear native failure
still wins. If those gates did not run, the report must say so. Qualification
must include honest long/max-length cross-H200/H100 controls and adversarial
small-coordinate proof mutations. Numerical UNKNOWN is never permission to
accept a forged proof. That future scientific contract requires separate ROOT
review and deployment; this historical overlay does not activate it.
