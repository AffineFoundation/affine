# Prospective live MATH rewards

The reward bridge is live. The first admitted public MATH epoch produced 42
audited points from 19 miners. Its encrypted commitment finalized, then the
revealed vector passed independent recipient, normalized-share and current-owner
checks at Finney block 9203767. Both legacy payout writers are disabled/inactive.
Historical pilot epochs are ineligible. Model execution and chain signing remain
separate roles; active weights do not establish individual wallet receipts.

The validator publishes a signed `live_reward_contract` in the first manifest
of each eligible epoch. It pins the approved source, checkpoint, penalty settings,
cutover identity and reward calculation. Compute jobs remain unable to submit
chain transactions. Only fresh `nonpayable-live-reward-math-v1-` compute epochs
with this explicit contract can produce separate `live-math-reward-v1-` records.
The compute `payable: false` flag preserves the execution restriction; the signed
reward contract declares reward eligibility. Existing nonpayable history is not
relabelled or passed to the historical payout reducer.

Verifiers use the bounded-random-v1 policy in [AUDIT_SAMPLING.md](AUDIT_SAMPLING.md).
The reward projector authenticates the first manifest, opening attestation,
registration snapshot, frozen receipt bindings, audit reports and final score.
It recomputes scores with the shared audit-only function. A credited batch must
have a fully audited success/failure pair for an authorized task and checkpoint.
Confirmed invalid batches apply the epoch's penalty multiplier; infrastructure
errors do not. Unchecked batches earn no points or training admission.

Observed cross-miner duplicate tasks earn zero. Sampled audits cannot establish
uniqueness against every unchecked claim, so reward records explicitly state
that duplicate coverage is incomplete. This first bridge requires the sampled
policy; full-audit reward contracts are refused before upload grants are created.

Hourly rewards sum exact adjusted points for epochs finalized in the completed
UTC hour. They use one million integer units per point and round only after
aggregation. Fresh on-chain UID/public-key ownership must match the signed
snapshot. The default reducer refuses the hour when an identity changes.
The live writer's signed operator policy now explicitly selects
`exclude-ineligible-v1`, starting with the pending hour ending 06:00 UTC on
October 4. Removed or changed identities are recorded in the signed hourly
proposal with their original identity, exact fractional points and reason.
Only currently eligible recipients enter the vector and its denominator. Earned
epoch scores and immutable reward records remain unchanged; excluded points are
not redirected to a recycled UID or automatically carried into a later hour.
The chain adapter independently checks every remaining recipient immediately
before submission. A further identity change during preparation still defers
that proposal. This fixes a pending hour blocked by three deregistered winners.
The original writer then reported successful submission and exited. Independent
readback at block 9208448 confirmed exactly 86 positive recipients, identical
normalized shares and current ownership. The next scheduled invocation advanced
to the following pending hour. These weights do not establish wallet receipts
or model improvement.

The operator exporter is invoked as `python -m ops.live_reward_exporter` with
private compute/reward state directories, the signed cutover anchor, validator
public key, private authority seed file and a fresh registration observation.
It writes authenticated reward records and hourly proposals; it does not submit
transactions. `ops.live_reward_submit.submit_hour` authenticates the proposal
before handing its integer points to the existing chain adapter.

The actual writer must hold the global owner/subnet lock, verify its own live
process identity, check the legacy validator's weight guard and establish that
both legacy burn and registration-weight timers/services are disabled and
inactive. Only that writer has the chain-signing wallet. Metadata-only test
receipts do not satisfy these operational checks. The public launch announcement
follows a verified chain receipt, not merely a successful dry run.

`python -m ops.live_reward_writer --cutover SIGNED-CONFIG.json --anchor
SIGNED-ANCHOR.json --authority PUBLIC-KEY` runs the single-writer path in dry-run
mode. Explicit `--execute` enables the chain handoff after operational cutover.
For prospective code upgrades, the signed cutover may include an
`approved_sources` map from archive SHA-256 to its exact local archive and signed
descriptor paths. Its primary `source` entry must remain in that map. The writer
authenticates every archive and selects the original module inventory pinned by
each epoch; upgrading the primary source never replaces older audit evidence.
The signed anchor must also approve a new source before its first epoch opens.
Without this map, the existing single-source contract remains unchanged.

An approval extension must also retain the original signed anchor for each
source in `approved_source_anchors`. Both completeness checking and reward
export select that source's original anchor. The extension preserves cutover
identity, effective time and all earlier source approvals. Previously exported
ledger records and closed-hour proposals stay byte-identical; changing the
anchor used to recompute old records is refused as an immutable ledger collision.

The runner derives actual process/boot identity, holds the global lock, queries
the old units and checks the reviewed production hook and suppression marker.
It reads original signed verifier requests and worker-authenticated completed
queue reports, checking source, runtime, frozen artifact and original expiry
before export. It uses current chain registrations rather than supplied identity
claims. Deferred hours retry the same hour. An uncertain exception after starting
a transaction requires chain reconciliation before retry; it is not automatically
treated as a failed submission.

Before closing an hour, the writer checks the actual controller finalization
watermark and requires complete signed sidecars for every original finalized
score. It repeats that check after chain identity discovery. An unresolved older
epoch waits instead of closing as zero. Records from an already closed hour must
already appear in its immutable ledger and signed proposal; delayed evidence
cannot silently lose rewards or create a second payment.

If an expired active epoch has neither an original score nor its signed sidecar
at the first completeness check, the runner reports
`waiting_for_epoch_finalization` with a successful timer invocation. It leaves
the reward cursor and prior handoff evidence unchanged and does not construct
a chain adapter, export an hour or submit a transaction. A lone score artifact,
missing finalized sidecars, inconsistent evidence or a finalization change after
chain identity discovery still refuses the invocation. This distinguishes normal
audit backlog from corruption without bypassing the closed-hour reward guard.
