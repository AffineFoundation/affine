# Measured learning and transport status — 2026-10-06

The live pilot trains on cheap-eligible, committed, unaudited task pairs. It does
not wait for inference audits or repeat inference verification in the trainer.
K=1/L=1 and the three-batch-per-UID cap remain the live contract. The matched
quota research runner is default-off; see [its protocol](MATCHED_QUOTA_RESEARCH.md).

The hourly weight writer now has a separate current-assessment policy rather
than a queue of finalized training epochs. It uses committed unique task counts,
cumulative authenticated audit estimates and immediate confirmed-invalid
penalties, with a six-hour contribution half-life. Training failures and an
unfinished opening are not scoring gates. Historical compute manifests retain
their original flags; the writer keeps a separate signed hourly assessment.
See [the exact calculation and submission controls](HOURLY_CURRENT_ASSESSMENT.md).

The unchanged independent 128-task cohort scored 77, 86, 84, 78, 85, 87, 83, 86 and 85 correct
on checkpoints 11 through 19. Checkpoint 16 gained five and lost three tasks
against checkpoint 15. Checkpoint 17 then fell to 83/128 on the same cohort. The signed summary and
archive hash were checked; all four original job archives were acknowledged and
the owned evaluator model retired automatically. Checkpoint 18 recovered to
86/128 on that unchanged cohort, with all four original archives acknowledged
and automatic owned-model retirement. Its signed summary and actual archive
hash were checked. Checkpoint 19 scored 85/128 (66.41%) on the same cohort;
all four signed original reports, actual exit-zero observations and full archive
readback acknowledgements were authenticated. This is fluctuation, not established convergence. Dataset size and additional samples alone do not establish learning.

Epochs 24 and 25 completed in 50.4 and 49.3 minutes. Epoch 25 trained on 256
distinct eligible task pairs, advanced optimizer step 15 to 16 and published
checkpoint `b2b68f9c15cc22cdd9ba812c7f79c83f65be22ae75102d957e685882cc6af66c`.
Its automatic trainer cleanup completed. Epoch 26 also completed in 49.4 minutes,
trained on 253 eligible unaudited task pairs and advanced optimizer step to 17.
It published checkpoint
`120bbf7416322d93e20502935272a6956edbc4c660f1029d07abc20f1d2c12da`
and automatically retired its previous local checkpoint. These three epochs do
not establish sustained hourly uptime.

## Optimizer bandwidth

The original epoch-25 training report contains 23 shard publication receipts,
totalling 91,387,491,264 bytes. Deriving durations from their original timestamps:

| Measurement | Seconds |
| --- | ---: |
| First shard start to last shard completion | 863.87 |
| First upload start to last upload completion | 825.91 |
| Median shard materialization and hashing | 27.06 |
| Median shard upload | 267.05 |

Eight transfers overlap. Summing shard durations is **not** epoch wall time.
The publication span implies approximately 111 MB/s aggregate upload throughput;
this is an observation from one run, not a measured provider bandwidth limit.
Independent full durable readback still follows before optimizer authority
advances. Removing that integrity check is not an optimization.

Sixteen streams require a new transport contract and actual RAM/disk admission;
they cannot be enabled by relabeling the existing eight-stream execution.
Any connection, compression or hardware change needs a bounded measured probe
before it is extrapolated to the full state. Whole-host-only rental offers are
not equivalent to an inexpensive single-GPU bandwidth upgrade.

A separate single H100 was rented at $1.30/hour for isolated qualification.
Its first bounded probe used eight 32-MiB random objects per profile, with payloads
held only in RAM. Serial PUT plus full GET took 24.05 seconds for 256 MiB;
eight concurrent transfers took 5.60 seconds for another 256 MiB. All 16 objects
passed a second independent ROOT full GET/SHA256 check, and the report and signed
ACK are durable in R2. This confirms the benefit of concurrency for those small
objects. It does not establish a full-state advantage over the live trainer,
which already uses eight streams. No training/model state was moved or changed.

## Training-data investigation

In the selected epoch-25 pairs, 121 of 256 negative trajectories reached the
1,024-token cap. All 120 pairs with reported reference margin above one had a
cap-length negative. This is a correlation in unaudited inputs, not a fraud
finding or an independently verified outcome. The controlled quota trial records
class supply, censoring, truncation and runtime before comparing updates from the
same parent on identical tasks and the same untouched held-out cohort.

## Token-only qualification

The first corrected-source H200 run reproduced three accepts and one numerical
unknown, and rejected all 80 token mutations in both official and token-only
direct checks. The 168 direct checks took 281.85 seconds: model prefill 20.35,
TOPLOC 0.69, native session initialization including hashing 103.38, and native
steps 9.85. These measured components do not exhaust total runtime.

That run subsequently failed in backend integration because its original opening
did not contain post-freeze audit receipts. The direct controls are preserved,
but the run is not a completed end-to-end qualification. A separately labeled
research post-freeze context is being tested; the live proof contract remains
selected-token logprobs plus TOPLOC. Two-machine qualification and fresh current
checkpoint/runtime admission must precede a token-only production cutover.

## Actual verifier recovery and remaining startup cost

After the WAL/FULL queue recovery, the first isolated-bootstrap canary completed
its original attempt-one lease. The worker-signed report was authenticated and
its digest matched the stored report. One batch passed the original v3 audit;
no chain transaction occurred. Its downloaded checkpoint was automatically
retired after the report acknowledgement, and the worker claimed another job.

Claim to report acknowledgement took 244.07 seconds. Checkpoint materialization
and authentication took 209.57 seconds for approximately 15.24 GB, and runtime
model construction took 17.26 seconds. Downloads remain serial across checkpoint
files in this scientific source. Bounded parallel authenticated downloads are a
prospective transport improvement, not a measured deployed speedup. Other workers
are being updated without interrupting active jobs; one canary does not establish
sustained fleet recovery.

Epoch 27 completed in 48.87 minutes, trained on 227 eligible unaudited task pairs,
advanced optimizer step 17 to 18 and published checkpoint
`703a7a310e06c9b4ca80fa872c28707d01ad6d64ea7d0a7ecd4bf52566cc2810`.
Its learner completion was authority-signature verified. Four consecutive
sub-hour learning iterations do not establish sustained uptime or convergence.

The corrected token-only matrix completed on H200 B: 182 control rows, 192 causal
forwards, zero cached fallbacks, genuine successful exit, full archive readback
and automatic owned-model retirement. Each direct mode reproduced three accepts,
one unknown and 80 CDF-invalid mutations. Backend native false-outcome, same-task
off-policy and invalid-dominance controls passed. Total wall time was 551.27
seconds including checkpoint download and loading. Official direct checks took
158.87 seconds and token-only checks 125.71 seconds in a single ordered run;
this is not a controlled speedup estimate. H200 A subsequently completed the
same 182 controls with matching verdicts, 192 causal forwards and zero cached
fallbacks. Its original process exited successfully; the complete archive passed
signed full R2 readback, and automatic completion retired all ten downloaded
model files (15.24 GB) before production resumed. Its total wall time was 540.92
seconds. The two-host comparison is now complete, including confirmed-invalid
dominance over numerical unknowns and native/quota checks. This qualifies those
research controls on the pinned checkpoint; it does not admit the new artifact
format for production or establish general cross-hardware numerical calibration.
The separate explicit token-only miner/backend contract is being integrated and
still requires fresh current-checkpoint qualification before deployment.

A bounded lossless-compression probe read one 4-MiB interior slice each from the
actual checkpoint-18 FP32 master, first-moment and second-moment tensors. No
optimizer/model bytes were changed or retained locally. Zlib levels 1, 3 and 6
all reconstructed the slices exactly; compressed sizes were approximately
90–93% of original size, with about 0.11–0.13 seconds compression per 4 MiB on
the coordinator. The probe report and signed acknowledgement were fully read
back from R2. These three slices do not establish a full-state compression ratio.
This small saving does not support introducing a compressed state contract
ahead of the measured parallel-download work.

The explicit default-off token-only contract is now implemented in the repository.
It removes the miner generation pass for log-probability arrays and TOPLOC
fingerprints, rather than merely stripping those artifacts after computing them.
New versioned token transport and cheap-training documents preserve historical
formats. Its prospective backend performs a causal prefill with prescribed draws
and a fresh native grade, while numerical ambiguity remains neutral. Authenticated
job-scoped native source validation avoids repeating full source hashes. CPU and
fresh-process source-admission controls pass; current-checkpoint GPU generation,
end-to-end transport and deployment admission remain required. It is not live.

An isolated H100 transport experiment authenticated the same ten checkpoint-18
files (15,242,726,226 bytes) in 218.10 seconds serially and 86.01 seconds with
four concurrent GETs, a measured 2.54x improvement for that ordered experiment.
Both complete file maps matched. Their durable timing files and failed original
job were fully archived and read back from R2; automatic retirement removed
both local model replicas afterward. The later scientific control failed because
the test harness omitted commitment/frozen-receipt context, so this is transport
evidence only, not model-verifier qualification or production activation.
A reusable CPU preflight now tests that exact commitment context before costly
model downloads, and individual control records are persisted as they complete.

All seven corrected production verifier roles have now completed authenticated
original reports and acknowledgements. The seventh role naturally settled its
prior lease before replacement; its old loader failures remain preserved. Its
first corrected report was followed by automatic obsolete model/input retirement
and a new admitted job. No active scientific process was killed or historical
failed attempt relabeled. A snapshot contained 74 successful signed report/ACKs,
224 valid and 19 worker-classified invalid outcomes; these counts are not an
independent fraud finding or universal assurance.

Epoch 28 subsequently completed in 53.38 minutes with 213 unaudited input pairs,
advanced the durable optimizer to step 19 and published checkpoint
`136c3f788eb20da982bc798ec1fb54cbc1463c6c7e1e6eeb2a4d3d8004ba0280`.
Its signed learner closure was authenticated. Five consecutive sub-hour learning
iterations do not establish sustained uptime or convergence; chain transactions
remain disabled.

The next opening, epoch 29, is held by numerical confirmation: its actual
log-probability error was 0.00075531005859375, above the proposed
0.0005950927734375; its CDF error remained within the proposed bound.
Both original calibration reports completed successfully and remain preserved.
Bounded, default-off confirmation recalibration now passes 42 CPU controls;
it uses authenticated failed measurements to propose a new bound and requires
a fresh confirmation, with fixed attempt/deadline limits and existing hard
maxima. It restored the live opening on October 6 after a genuine fresh calibration and
independent confirmation. Epoch 29 reopened for 247 registered identities and has reached the training
stage with 201 eligible unaudited pairs (237 committed). It runs under source `f21373d7ccb167bcd868f5ed03ce9ac7ef567d894e8ecc9e61f5d7f8645b67b8`,
with the original checkpoint and optimizer step 19 preserved. The new bound is
0.0016632080078125; actual confirmation measured at most 0.00029754638671875.
Opening is not evidence that epoch 29 training or convergence has completed.

The legacy pod reaper released the explicitly retained isolated H100 on October
6 at 16:13 UTC after incorrectly classifying it as ownerless. The provider
release ledger confirms the deletion. An exact retained-name ownership rule
has now been installed and tested through the actual reaper functions; other
abandoned pods keep their existing release rules. This protects the replacement
qualification rental without disabling automatic cleanup. All original research
archives remain durable in R2. Production verifier cache retirement remains
automatic after acknowledgement and respects active leases.

## Durable CPU restart recovery

The independent coordinator API, continuous auditor and ordinary learner now
run as boot-enabled systemd user services with automatic restart on failure.
Their durable ROOT-signed execution policies pin the same scientific source,
configurations, authority, source admissions and queue. Policies do not bind
a temporary startup PID or expire after the initial handoff. Scientific job
and upload deadlines remain unchanged.

Actual controlled restarts of all three CPU services passed without policy
renewal. The original queue inode, WAL/FULL mode and schema were preserved.
Exactly one epoch-29 training job remained; its original remote runner and
child continued unchanged. At the last observation epoch 29 was still training
201 eligible unaudited pairs, with committed optimizer step 19. An in-progress
StateCache is not a completed checkpoint or durable optimizer commit.

See [startup policy and guards](DURABLE_AUDIT_SERVICES.md). A hardware reboot
was not performed; boot enablement and controlled service restart were verified.

## Epoch 29 original completion and checkpoint 20

The original epoch-29 job completed with 201 eligible unaudited task pairs,
advancing optimizer step 19 to 20. The independent reader authenticated all
23 optimizer shards, totalling 91,387,491,264 bytes. The signed completion,
checkpoint publication and durable authority pointer agree on checkpoint
`94e8999b75838cddf70d3e767318e7b169a5421c890ea9c7eeef2e2aa6104e6d`.
Round 30 is training its original job. The independent SAME128 evaluator completed
checkpoint 20: 84/128 (65.625%), versus checkpoint 19 at 85/128 and checkpoint 18
at 86/128. The authenticated ROOT summary binds the unchanged cohort hash
20a077180d7cc088669f53bd551c5bf3c4aa51ded6367463f405094ad054b153
and all four original full R2 archive acknowledgments. This does not show convergence.

The original controller recorded 301.06 minutes total, including 234.51 minutes
of opening/calibration recovery. After reopening, collection through publication
took 66.55 minutes, of which training/publication took 54.78 minutes. The original
training report records 996.05 seconds of cold optimizer restore, 432.33 seconds
of training/checkpoint work and 1,105.46 seconds of state export/upload; these
components do not exhaust controller wall time. Source-change cache retirement
was acknowledged automatically. A faster subsequent warm restore must be measured,
not assumed. This completion clears the publication blocker but does not prove
continuous hourly operation or held-out convergence.

## Epoch 29 learning-data measurements

The authenticated original training report records 201 task pairs, one full-model
update and 339 gradient tensors. Reported training-pair margins increased for
180 of 201 pairs, with a mean increase of approximately 0.017. The initial
centered preference loss of 0.693147 is expected: this epoch's reference margins
are computed from its initial current model. It is not evidence of zero gradients.
Training-pair improvement is not independent held-out improvement.

Length remains a significant confound: 125 of 201 submitted negative trajectories
reached the 1,024-token cap, versus two positives. Median negative length was
1,024 tokens; median positive length was 287. These are measurements of eligible
unaudited inputs, not independent certification of native outcomes or provenance.
A preference update may learn completion-length differences as well as mathematical
skill. Increasing quota alone must therefore be tested against distinct class
supply, truncation and the unchanged held-out cohort. The planned same-parent
1P1N/2P2N comparison preserves identical tasks, optimizer parent and update settings;
its first generation-only stage measures 16 prescribed attempts on each of two
predeclared tasks. No new quota or learning-rate policy is activated by these
observations.

## Epoch 29 capture coverage, separately from audit coverage

A later authenticated inventory distinguishes 537 timely proof batches across
184 committed miner identities from the 237 training documents actually captured.
The capture step deferred 300 slots as infrastructure under its four-reader,
deadline-plus-60-second budget. R2 server upload metadata shows those 300 token
objects were present within the signed submission window. No structural exclusion
occurred among captured candidates; 36 duplicate-task exclusions left 201 eligible
training pairs. Earlier references to 237 committed pairs describe this captured
training candidate set, not the full proof submission population.

The continuous auditor completed 135 original grouped jobs covering all 537
unique proof rows: 528 accepted and nine TOPLOC rejections across three identities.
No batch was double-counted. Reward evidence deduplicates the original population
row before updating confidence or penalties. Two of the 201 unaudited training
pairs were later TOPLOC-rejected; that does not alone establish intent. Conversely,
329 audit-accepted proof rows were outside the learner-eligible set, primarily
because their token documents were deferred by bounded capture.

A prospective versioned capture policy is being prepared to improve parallel
capture without extending submission deadlines, rewriting old deferrals or
changing current scientific jobs. It has not been deployed. Independent TOPLOC
reference replay remains pending. Held-out checkpoint 20 completed at 84/128.


## Epoch 30 recovery and sample-supply control (2026-10-06, 22:33 UTC)

Epoch30 genuinely trained79 unaudited task pairs and339 gradient tensors from
optimizer20 to21. Original model publication failed with a transport EOF; a
bounded publication-only recovery committedCP21 after independent full model
and optimizer readbacks, with no retraining or lineage reset. Automatic cache
promotion completed and the recovery process exited normally. The ordinary
capture-v2 learner has startedE31, but its improved capture throughput is not
yet measured. E30 captured91 of549 declared timely batches from192 identities;
12 task duplicates left79 training pairs. Infrastructure deferrals are not fraud.

A separately scoped CP19 supply experiment completed32 prescribed attempts
across two predeclared tasks, with no optimizer updates. Task193 produced16
successes and no failures. Task28 produced3 successes and13 failures, all13 at
the1,024-token cap. One success/failure pair was available by attempt3; two of
each only by attempt9. These two tasks are too narrow to justify changing the
production quota. Truncation remains a training confound, and neither supply
nor a successful publication demonstrates held-out convergence. CP20 SAME128
remains84/128; CP21 evaluation is not yet verified here.

The CP21 fixed32 diagnostic subsequently completed at19/32 (59.375%). ROOT
verified the original signed job, report digest and durable full-readback ACK.
The independent SAME128 experiment is still running its original four-group
sequence; its result cannot be inferred from this smaller diagnostic.


## Epoch 31 measured capture recovery

The signed eight-reader capture-v2 policy retained all574 timely batch documents
from194 authenticated miners, with zero deferrals.146 duplicate-task exclusions
left428 eligible unaudited pairs. The bounded postfreeze selector chose256 for
the original training job, retaining172 unselected pairs as audit/reward eligible.
This is not a claim that574 pairs were verified or trained. Token capture took
27.583 seconds, finishing deadline+36.839 seconds within the signed+60 budget.
Whole capture took74.800 seconds including final commitment publication; cheap
reads/decode22.779 seconds, complete collection/selection98.300 seconds. E30 had
retained91/549 and trained79. More usable inputs are now reaching training, but
new checkpoint completion and held-out gains remain to be measured.

## Untouched-base control: partial matched evidence (2026-10-07)

The first three original groups of the wider held-out evaluation completed;
their authenticated native reports give the untouched base 87/96. These are the first
96 tasks in the frozen group order, not a complete 750-task result. Fifteen tasks
overlap the completed CP24 SAME128 experiment with exactly matching native task
hashes, sampling seeds and rendered prompt hashes. On that overlap the base
scored 15/15 and CP24 scored 12/15: zero gains, three losses, twelve unchanged.
The loss indices are 90, 429 and 949. Original source, grader, environment and
sampling harness bindings match; the two observations used different GPU hosts.

This is a provisional regression signal, not a complete learning comparison or
proof of its cause. The original same-H100 all750 comparison remains in progress;
its complete paired result must not be inferred from these early groups. A
separate same-parent, persistent-Adam controlled experiment examines capped
versus completed-EOS negative trajectories on eight identical tasks. Its original
training job is still running and no outcome is claimed. Production learning
continues independently; published checkpoint counts and in-sample margins do
not establish held-out convergence.
