# Prospective live MATH rewards

The reward bridge is implemented; activation still requires admitted GPU source,
a fresh public epoch and an actual verified chain submission. Historical pilot
epochs are ineligible. Model execution and chain signing remain separate roles.

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
snapshot. Changed ownership refuses the hour instead of redirecting rewards.

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
The runner derives actual process/boot identity, holds the global lock, queries
the old units and checks the reviewed production hook and suppression marker.
It reads original signed verifier requests and worker-authenticated completed
queue reports, checking source, runtime, frozen artifact and original expiry
before export. It uses current chain registrations rather than supplied identity
claims. Deferred hours retry the same hour. An uncertain exception after starting
a transaction requires chain reconciliation before retry; it is not automatically
treated as a failed submission.
