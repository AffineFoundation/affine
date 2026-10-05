# Superseded operational observations, October 5

These entries describe earlier observations. Consult public llms.txt and the latest signed manifest for current status.

Operational update, 2026-10-05: E15's sole original training job finished
successfully with 240 unaudited eligible task pairs and one optimizer update
from step five to six. The trainer did not repeat inference verification.
The new state is staged; the controller must finish independent durable
readback and publication before committing step six or opening E16.
The physical training job took 60.49 minutes: 17.58 minutes restoring the
FP32 optimizer, 8.70 minutes training and checkpointing, 18.48 minutes uploading
the FP32 successor, and about 15.72 minutes of startup and input overhead.
This epoch therefore does not meet the one-hour target. More verifier nodes
alone cannot shorten that training/publication critical path.
A new scoped controller is active, adopting the same original job using the
reviewed detached transport. No duplicate training job was issued.
Six verifier workers remain active. The replacement H200 passed full source,
runtime, checkpoint hashing and a finite GPU forward; fresh sampling/TOPLOC
controls are in progress before enrollment. It is not yet an active verifier.
The matched E14 diagnostic remains 20/32 before and 20/32 after. A separate
128-task cached native evaluation is prepared to increase measurement coverage;
no sustained learning or convergence is claimed.

Verified recovery update, 2026-10-05 19:07 UTC: the local E15 finish-once controller
failed when its original SSH launch command timed out after 1,800 seconds.
The same remote trainer remained live and progressed through optimizer restore
into computation; no second training job was issued. The independent queue API
was restored on the same SQLite database and worker roster, and the continuous
auditor completed fresh scheduling ticks. An original-job-preserving controller
recovery is being prepared; optimizer step six remains uncommitted.
The prospective launch transport now detaches descriptors explicitly and probes
the original job after a lost reply, with exclusive duplicate prevention. Five
transport controls and 24 existing remote controls passed. This operator repair
is not yet activated in the failed controller's immutable launch tree.

Verified fleet update, 2026-10-05 18:57 UTC: the two disk-full verifier workers are
quarantined. Their exact original processes were stopped only after checking
PID/start time, command and absence of live children. No checkpoint, rollout,
queue history or tunnel was deleted or changed. Six other verifiers continue.
One replacement H200 has been rented at USD 5.76/hour with actual 1.287 TB free
disk; runtime and source qualification are still pending, so it is not counted
as an active verifier. A second rental was refused because that offered node
already has a broken pod; that pod was preserved. Replacement capacity is
being qualified before any new worker joins the queue.
E15's original trainer has progressed into GPU computation after restoring its
original optimizer parent. There is no committed step-six checkpoint yet.

Verified operational update, 2026-10-05 18:52 UTC: E14's original paired held-out
evaluation finished: 20/32 correct before and 20/32 after, with no infrastructure
failures, one new win and one new loss. Original signed jobs, checkpoint bindings,
report hashes and the exact task/seed order were checked. This is no net held-out
improvement, and the fixed 32-task diagnostic does not establish convergence.
E15's sole original trainer is still restoring optimizer parent five; requested
step six is not committed yet. Two verifier workers have confirmed disk-full
infrastructure failures during checkpoint download, before inference or reports.
The other six have complete current-checkpoint caches and available disk. These
failures are not evidence of miner cheating; quarantine/replacement is being
prepared without deleting local history.

The prospective native environment-session reuse source has passed actual
full-file/runtime qualification on all eleven existing role endpoints, bounded
real-native reset equivalence, and 687 CPU controls. It is not activated for E15.
A separate opt-in cached evaluator is prepared for actual two-checkpoint hardware
qualification; its future diagnostics are explicitly native-graded, not miner
proofs. Neither prospective change alters the running epoch's source or contracts.

Current operational update, 2026-10-05 18:37 UTC: E15's mining window is closed.
Collection produced 270 candidates and 240 cheap-eligible unaudited inputs.
Its sole original training job started at 18:22 UTC on the qualified
higher-memory H200 that previously served as the owned miner. The previous
trainer is now assigned the miner role; physical checkpoint-cache ownership
was remapped without deleting state or changing scientific contracts.
The new trainer has about 309 GB usable RAM and 130 GB available disk, exceeding
the unchanged streaming-worker reserves. Checkpoint CP76bc, optimizer parent
five and the signed four-stream E15 contract are preserved. The requested
next optimizer step is six; it is not committed yet. Latest observed progress
was input loading before optimizer restoration, not a completed training update.
Eight verifiers continue independent audits. Held-out evaluation remains
asynchronous. Do not submit to the closed E15 window; read the next signed
manifest when opened. The local controller is configured to finish this same
epoch once, allowing reviewed future changes only at the next epoch boundary.

The real 18:00 UTC E14 reward snapshot now has 77 nonzero normalized weights,
totaling one. These are estimated valid contributions under the continuous-audit
policy, not verification of every submitted sample. The signed snapshot and
hourly aggregate were authenticated, recomputed and read back exactly from R2.
Chain transactions remain disabled. E15 has no completed training update yet
and is not included in that hourly reward record.

Deployment update, 2026-10-05 17:48 UTC: epoch
nonpayable-live-reward-math-v1--1791222075-15 is mining from checkpoint
76bc599baf173cb4d071ff1959492c0a3c23377143aa2d69b749c17becc34785
and actual committed optimizer parent five. Its original ROOT-signed mining job
and nested manifest preserve checkpoint, optimizer and genesis history. The
signed mining window is 17:44:29–17:54:29 UTC and uses checkpoint-specific
calibration. Consult mining.json and the signed manifest for the current phase;
do not upload after its deadline or against an earlier closed window.

The preceding E14 trainer completed successfully on 161 unaudited task pairs,
with finite loss/gradients and changed model weights, without sample
re-verification. Independent full-state readback and checkpoint adoption made
optimizer step five durable. E14 took 64.795 minutes from its signed manifest
to closure, or 100.71 minutes including opening recovery. The one-hour target
was missed. The fixed 32-task comparison finished at 20 correct before and 20 after, with
no infrastructure failures and one win/loss exchange. There is no net held-out
gain; convergence remains unproven.

E15 retains its signed four-stream optimizer transfers. The replacement trainer
passed actual four/eight-stream resource qualification without reducing reserves.
Eight-stream transfers are prospective and require the next signed epoch contract.
Eight audit workers and the four-thread continuous auditor remain independent
of training and evaluation. This remains a nonpayable pilot: an opening is not
a completed training update, a reward snapshot or a chain weight transaction.

The audit coordinator was restarted with an operator-only repair on October 5:
it now authenticates signed learner closures and ignores their unsigned local
mirrors, avoiding duplicate reads and a completion-timestamp parsing crash.
The original closure times, science, sampling contracts and submitted artifacts
are unchanged. A fresh scheduling tick completed at 17:52 UTC, enqueuing eight
jobs that verifier workers have leased. This is scheduling progress, not a claim
that all eight have completed or passed verification.

