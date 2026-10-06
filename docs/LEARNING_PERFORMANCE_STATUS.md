# Measured learning and transport status — 2026-10-06

The live pilot trains on cheap-eligible, committed, unaudited task pairs. It does
not wait for inference audits or repeat inference verification in the trainer.
K=1/L=1 and the three-batch-per-UID cap remain the live contract. The matched
quota research runner is default-off; see [its protocol](MATCHED_QUOTA_RESEARCH.md).

The unchanged independent 128-task cohort scored 77, 86, 84, 78, 85 and 87 correct
on checkpoints 11 through 16. Checkpoint 16 gained five and lost three tasks
against checkpoint 15. This is an encouraging rebound, not established
convergence. Dataset size and additional samples alone do not establish learning.

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
