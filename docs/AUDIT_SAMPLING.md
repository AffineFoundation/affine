# Bounded audits and proposed weights

The GPU controller accepts an opt-in `audit_policy` using the contents of
`configs/bounded-audit-policy.json`. Existing configs default to full audits;
this change does not alter a running or signed epoch, deploy workers, or enable
chain transactions. Deploy the same new pinned source to controller and workers
before opening a new epoch with this policy. The legacy CPU controller is not
the bounded sampling coordinator.

Set `max_batches` in the GPU controller config to the per-UID epoch submission
limit (1–256, default 3). Existing signed transport, decompressed tensor,
rollout-token and K/L limits remain enforced; increasing the batch limit does
not increase the artifact byte limit. Each authenticated subnet identity gets
one quota. This does not prevent someone registering multiple identities.

After freeze, the controller generates and persists a random challenge bound
to all frozen receipt hashes. It allocates at most `epoch_budget` batch checks:
first a randomized minimum allocation per miner, then random remaining slots.
When the budget cannot cover everyone, some miners get zero expensive checks.
Allocation uses the signed per-miner batch cap as a conservative population
bound, not miner claims or upload bytes. Unused slots from smaller submissions
are not counted as checks or automatically recycled in this first implementation.
Allocation uses a prefix-sum tree: memory grows with the number of miners, and
each remaining audit slot takes logarithmic work rather than scanning every miner.
This retains the existing seeded selection and allocation policy.
Worker reports contain actual selected batches and assurance separately. Byte-identical
cross-UID submissions get no expensive audits or points; this also prevents
a shared artifact hash from multiplying a single allocated audit slot.

All submissions undergo bounded decoding and batch structure/quota checks.
Only selected batches get model/TOPLOC/environment verification. Selection is
a deterministic prefix of a post-freeze random permutation; retries keep the
same selection. Independent existing verifier workers claim the signed jobs
using the existing coordinator leases. Add verifier workers to increase
throughput; the controller's budgets remain explicit and bounded.

Confirmed invalid data can trigger expanded checks, up to `escalation_budget`
and `maximum_per_miner`. Expanded jobs recheck initial selections too; those
repeated checks are charged to the escalation budget. Original reports are
retained, and inconsistent repeated outcomes stop finalization. Infrastructure
exceptions are not evidence of fraud. Explicit false verification or typed
`InvalidSample` comparison failures are evidence of invalid submitted data.
Unexpected exceptions during sampled inference verification fail the job for
ordinary worker retries, without generating a fraud report. Other unclassified
validation errors earn no batch points and remain visible for diagnosis. They do not imply all unchecked data is valid.

Proposed weights are a pure function of authenticated audit reports and penalty
parameters:

    adjusted_points = unique_fully_verified_task_points * multiplier ** invalid_batches
    weight = adjusted_points / sum(all_adjusted_points)

The shared `subnet.scoring.adjusted_point_fractions` function computes exact
rational adjusted points. Preview scores and the payout bridge use this same
arithmetic. Hourly payouts aggregate adjusted points before integer rounding;
they do not round each epoch separately or sum normalized epoch weights.
Publish penalty settings in the signed epoch manifest before mining starts.
Changing an operator config applies to future epochs, not frozen audit history.

`invalid_batch_multiplier` defaults to 0.5. `zero_epoch_after` defaults to 0
(disabled); a positive value zeros that miner's epoch points once that many
confirmed invalid batches are found. `penalize_structural` defaults to false.
There are no wallet confiscations, permanent bans, or historical penalties in
this policy. All-zero points produce an empty proposed weight map.

Unchecked batches earn no points and cannot enter training. The trainer uses
the final signed audit allocation and rechecks exactly those selections. A
sampled result is marked provisional because unchecked duplicate claims remain
unresolved; it is not an estimate that every batch is correct. The existing
balanced replay pool requires historical full audits, so leave `balanced_replay`
disabled for a sampled pilot until replay admission is separately qualified.

To independently recompute weights, create a validator-signed ledger whose
payload includes `epoch_id`, `payable: false`, `receipts` keyed by miner and
`reports` keyed by miner. Every report is itself validator-signed and binds its
`epoch` and `submission_sha256` to that frozen receipt. Run:

    .venv/bin/python -B -m ops.audit_weights --ledger LEDGER.json \
      --authority VALIDATOR_PUBLIC_KEY --policy configs/bounded-audit-policy.json \
      --output NEW-proposed-weights.json

The command verifies signatures and receipt bindings, refuses overwriting an
existing output, and never submits chain weights. Budget numbers in the example
are configurable starting values, not measured verifier throughput guarantees.

To operate the sampled controller, add `audit_policy` from the example config
and set a fixed `max_batches` per UID. Run additional verifier workers against
the same coordinator to increase capacity; the existing job leases, retries,
and signed result acceptance handle concurrency. The weights process reads
authenticated final reports and the epoch's penalty settings; it does not run
inference or inspect miner upload volume to create additional points.
