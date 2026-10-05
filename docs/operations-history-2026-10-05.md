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



## Public guide historical contract material, archived 2026-10-05

# Historical operational records and earlier contracts

The records below describe earlier deployments. They do not override the
current signed epoch manifest or the deployed contract described above.

Operational update, 2026-10-05: E12 opened at 06:32:43 UTC as
nonpayable-live-reward-math-v1--1791181963-12 using approved source
a65d7ed87f036f286bcc374ed6713f29c6ed4e3141fc1e5dd4ec9948962f0f0f.
It preserves E10's learned checkpoint
6a2bb631eebdc748b577976038a673437309d7953d3440eb8e873e5035671ea7
and persistent optimizer step three; no optimizer reset is authorized.
The original E10 matched 32-task diagnostic changed from 19/32 to 21/32.
That small measurement is not sustained convergence or a one-hour epoch.
Check mining.json and the signed manifest for current phase and availability.
E11 closed without a training update after infrastructure recovery: inline freeze copies
exhausted its metadata budget, and the coordinator's legacy whole-object hash
gate could not admit selected child artifacts. Its audit window expired with
no queued audits and no accepted training samples. The original no-update
closure also exposed a legacy-only history renderer. E12 includes the corrected
queue, capture and history paths. Deferred uploads are not fraud. The learned
checkpoint and optimizer step remain unchanged; no successful E11 update or
reward is claimed.

The approved a65 recovery code requires the uploaded signed
commitment's raw JSON bytes to use the canonical serialization produced by
the official miner CLI. Its child audit jobs bind the original commitment,
slot, task, artifact hash and size; malformed bindings are refused before
queue admission. Capture reads all discovered small commitments before large
conditional copies. If metadata capture remains incomplete at the signed
cutoff, a separate signed infrastructure-skip record preserves the partial
journals and advances without fabricated receipts, audits, rewards or training.
These changes do not rewrite E11's original frozen compute source.

Prospective lossless packing code on GitHub main adds a signed
artifact_compression_policy (lossless-deflate-v1) and official CLI support.
E12 does not enable it and retains historical DEFLATE level six. A measured
existing 1.17 GB uncompressed pair packed in 16.1 seconds at level one versus
145.5 seconds at level six, with 13.2% more compressed bytes and identical
uncompressed tensor/metadata hashes. This CPU benchmark is not an observed
production epoch speedup. Original artifact and size limits remain unchanged.
Details: https://github.com/AffineFoundation/affine/blob/main/docs/LOSSLESS_ARTIFACT_PACKING.md

Prospective asynchronous owned mining is also on GitHub main. For signed
bounded-hourly-phases-v1 epochs, it records one original miner job and enters
collection without waiting for its terminal report, so submission capture starts
at the signed deadline. This is not active in approved source a65 above.
A later job cannot reuse that physical miner until the original supervisor and
child have genuinely finished; missing or unreachable reservations block dispatch.
External miners, generation deadlines, upload expiry, quotas, forced sampler,
TOPLOC, native grading, audit admission and penalties are unchanged. The intended
next source requests lossless-deflate-v1 level one in its signed manifest. Miners
follow that policy automatically; --compression-level 1 is only an assertion,
never a way to override the manifest. Existing prepared slots keep their exact
acknowledged bytes. Neither prospective change retroactively alters an epoch.
Details: https://github.com/AffineFoundation/affine/blob/main/docs/HOURLY_OWNED_MINER_DISPATCH.md

E11 activates small signed commitments plus separate immutable pair objects,
600 seconds for mining, 12 initial random expensive audits plus four reserved
escalation audits, and four qualified verifier workers. Full exact sampler,
TOPLOC and native grading remain mandatory for selected pairs. Only accepted,
fully audited pairs earn points or train. One confirmed invalid batch zeros
that miner's epoch score; infrastructure failures and unselected submissions
are not fraud. Audit budgets and deadlines do not guarantee hourly completion.
Details: https://github.com/AffineFoundation/affine/blob/main/docs/HOURLY_EPOCHS.md

The trainer uses authenticated compact audit inputs without repeating model
verification. E11 requests one actual persistent update and four optimizer-state
transport streams. upload-only-independent-full-v1 removes the trainer's
redundant full shard download. A qualified reader on a separate physical machine
must fully download and SHA-check every state shard. Checkpoint byte staging
can overlap that readback, but both authority publications and reward readiness
wait for validated original receipts and successful terminal evidence.
The independent evaluator observes committed checkpoints outside the epoch's
training critical path. E11 has not yet completed training or publication.
Details: https://github.com/AffineFoundation/affine/blob/main/docs/UPLOAD_ONLY_INDEPENDENT_STATE_READBACK.md

The active durable-training-before-reward-v1 policy holds chain eligibility
until signed readiness binds audited scores, the original training job,
advanced persistent lineage and durable checkpoint/state publication. The GPU
compute manifest remains nonpayable by design; the separately authorized single
reward writer handles eligible live rewards. A successful writer invocation
alone is not an epoch reward or chain-inclusion receipt.
Details: https://github.com/AffineFoundation/affine/blob/main/docs/DURABLE_REWARD_PUBLICATION.md

Dashboard: https://affine.io/
Live mining discovery: https://affine.io/mining.json
Measurements: https://affine.io/network-data.json
Code: https://github.com/AffineFoundation/affine (main)
Signed payout eligibility policy: https://affine.io/reward-policy.json

## Sampling cutover

Cutover state: sampling enforcement began with epoch nonpayable-live-reward-math-v1--1791128338-9. The current native-grader, compact-input and persistent-optimizer contract is active from epoch nonpayable-live-reward-math-v1--1791154261-10. Check mining.json and the signed manifest for current upload availability.

The current immutable source bundle is SHA256 5bcb01b6ca4187e6dce785c67256688faa950058c39dc7db70ce5b24e03e016f, admitted on all seven role machines. E10 used historical source7630e0d25f388fe480341614c861fa5be1650f110b49e343dc9134c05e492ab0; those jobs are unchanged. It preserves the forced-sampler computation and adds pinned native grading, authenticated-verifier-compact-inputs-v2 and persistent FP32 training. Historical E9 inference used e415623e5a8017133ff7bef3925b724441862f34b6d034ed0deb6b731a26e657; its receipt trainer used 94ff74eb335e24d4702da2ec10cc0aee068076b003e6c0b13b0e81f5090bc79c. Every epoch's signed manifest and source are authoritative. GitHub main may contain prospective optimizations; do not substitute a checkout for the signed bundle. mining.json closes uploads during audits, training, evaluation or controller absence.

The new forced-inverse-cdf-replay-v1 contract binds epoch randomness, checkpoint, environment/task, bounded attempt number, turn and token position. Miner identity does not change the draws. The signed attempt budget is 128: attempt numbers 0 through 127. Miners can search these attempts for one success and one failure, but cannot select an arbitrary seed, substitute answers, change temperature/top-p or change the sampler. The approved v1 harness is text-tools-long-v2: uncached eager autoregressive computation, at most 1,024 output tokens per turn, temperature 0.8 and top-p 1. The signed manifest and source take precedence over this guide.

An expensive audit regenerates the complete audited output and requires exact token and stopping agreement, then independently checks full conditional probabilities, TOPLOC fingerprints and native grading. Wrong sampler picks have no match-rate or numerical-tolerance exemption. This establishes consistency with the prescribed computation and sampler; it does not prove historical execution or unbiased selection of successes and failures. TOPLOC without this replay cannot distinguish sampled outputs from copied answers passed through the model afterward.

Qualification used real H200 generation: one successful 746-token output and one failed 1,024-token output, independent replay with zero honest false rejections, and three rejected attacks. A copied correct answer with newly computed genuine probabilities and TOPLOC proofs passed the old verification but failed the new sampler check. A separate H200 verified the genuine pair through the production signed-job path, and the trainer independently re-audited it and performed a full-model covered update. These are qualification controls, not a completed new public epoch or a learning-gain claim. Details: https://github.com/AffineFoundation/affine/blob/main/docs/FORCED_SAMPLING_QUALIFICATION.md

Epoch eight completed under its original rules; its final training was explicitly skipped after a disk-capacity refusal. E9 then completed three covered-v3 updates and published checkpoint 28d478ac16c36fb0ab816f0273aee220e42927136c1e5a8ee3c34792b6d38384. E10 begins from that checkpoint with an explicitly authorized persistent-optimizer genesis at step zero. Historical updates and private qualification steps are not persistent public optimizer counters. Old manifests, scores, evidence and writer history are preserved. The old burn and equal-registration writers remain disabled; the single guarded reward writer uses the signed policy for all eligible registered subnet identities.

