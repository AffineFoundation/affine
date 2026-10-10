# Start an Affine MATH miner

Read https://affine.io/llms.txt first. The latest signed OPEN manifest and its
approved source are authoritative; GitHub main includes default-off research.
Hourly weights use the best current authenticated miner assessment, retaining
the last valid assessment during evidence outages. Assessments and audits remain
independent of training; follow the current signed policies.

The current completed-answer contract requires eight distinct rollouts per task batch:
four native-graded successes and four failures, up to the signed `max_batches`
task batches per UID per epoch, and attempt nonces 0–999. Follow the signed OPEN manifest; historical
openings retain their original quotas and attempt ceilings.
Upload selected-token log
probabilities and TOPLOC, not full-vocabulary arrays. Audits check prescribed
sampler consistency; TOPLOC alone is insufficient. Training uses cheap-eligible
unaudited submissions independently of the audit loop. Token-only and three-way
research checks are not the active miner contract.

## Identity and setup

Any current subnet-120 identity with valid signed Affine Ed25519 activation can
enter the next eligible epoch; there is no operator whitelist. Registration,
activation and the epoch identity snapshot are separate. For an existing
subnet-registered Ed25519 hotkey, preview activation locally:

```bash
python -m subnet.register --wallet YOUR_WALLET --hotkey YOUR_ED25519_HOTKEY
```

Add `--execute` only when ready to publish that signed activation commitment.
This does not purchase a UID. Keep the hotkey seed private; do not give anyone
a coldkey, mnemonic, permanent bucket credential or decrypted upload capability.

Clone https://github.com/AffineFoundation/affine and install the repository.
Read [MATH_MINER_BOOTSTRAP.md](MATH_MINER_BOOTSTRAP.md) for signed-source admission.
Use the exact hardware, packages, native grader dependencies and numerical
profile pinned by the current manifest. Current qualified inference is H200/SM90,
FP32 eager, TF32 disabled, with torch 2.14.0, transformers 5.14.1 and toploc 0.1.6.
Package versions alone do not qualify another hardware/runtime profile.

## Run the signed source

Update main with `git pull --ff-only`. The continuous transport supervisor reads
https://affine.io/mining.json and authenticates the signed opening. The key file
contains your activated 32-byte Ed25519 seed as hexadecimal, mode 0600. Stop any
old miner using this state before starting:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B -m subnet.miner_supervisor \
  --discovery-url https://affine.io/mining.json \
  --authority 3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd \
  --source-cache /private/affine-approved-source \
  --key /private/miner.seed \
  --state /private/affine-miner-cache \
  --env-id affine_math --search-budget 32
```

Batch-cap update activated by signed epoch `nonpayable-live-reward-math-v1--1791602071-110`:
up to nine task batches per UID and up to 512 distinct training tasks per epoch,
with the same four-correct/four-wrong contents. The actual signed opening controls
capacity. Update and restart the transport supervisor once, omitting
`--max-batches` to follow each signed cap. An explicit `--max-batches 3` remains
valid and limits that client to three batches. Extra capacity is optional and
does not extend the upload deadline or earn duplicate-task contribution points.
The training cap is a maximum, not a guarantee that every batch is selected.

Signed epoch `nonpayable-live-reward-math-v1--1791622696-116`, starting
2026-10-10 at 09:02:08 UTC, activated `training_representative_policy`: the learner
can select one native-valid batch for a task shared by several miners, up to 512
distinct tasks per epoch. This changes training intake only: every colliding
submission still receives zero duplicate-task reward points, and the public
sampling, proof and upload contract stays unchanged. Miners following the signed
source need no special action. Earlier openings retain their original selection
rules; see [the learner design](COMMITTED_UNAUDITED_LEARNER.md).

Client update status (2026-10-10): the active signed source supports persistent
partial search groups and fresh remaining nonce attempts across same-state
restarts. Authenticate the actual source archive in the opening; a GitHub update
alone does not change the approved client launched by the transport supervisor.

The supervisor continuously follows epochs and authenticated source upgrades,
resumes bounded checkpoint transfers and verifies full file hashes before
launching the untouched approved miner. It removes only its own obsolete model
caches after admitting the successor. Reserve space for active and incoming
checkpoints. A per-state lock prevents overlapping children after restarts.

Thirty-two is the supervisor's default per-call search budget. For the active
v5 miner, `--search-budget` accepts 1 through the signed `max_attempts` (currently
1,000); the standalone CLI default remains 50. A later search call or explicit
restart using the same private state continues with unused nonces and retains
partial successes/failures for that task. It never starts another 1,000 attempts:
0–999 is the total nonce range per miner/task/epoch. Stop at the signed deadline.
Historical approved clients retain their original behavior and limits.
For the current contract, collect four distinct successes and four distinct
failures with eight distinct nonces. Only a complete group is uploaded.

Search progress is private local cache, bound to the exact signed manifest and
miner. Keep the same `--state` directory to resume; stop the previous process
first. A local lock prevents concurrent use of that search cache. This is not
cross-machine nonce coordination. At most 256 partial task groups and 64 MiB of
compressed partial proof data are retained; eviction discards old partial
examples but preserves consumed nonce cursors. The journal is capped at 96 MiB
plus a bounded SQLite rollback journal. Authenticated next-epoch handover retires
partial proofs and reclaims their local pages automatically. Completed upload
artifacts continue using the existing durable-resume path. The supervisor's
one-launch-per-epoch policy and task ordering are unchanged; this update does
not automatically relaunch an already-issued epoch.
The bootstrap verifies discovery, source and checkpoint hashes, then runs that
approved source. Signed code approval is not a sandbox. Miners need no permanent
R2 credentials; the manifest supplies encrypted private upload capabilities.

Keep the supervisor running; it launches each OPEN epoch only once. Never silently
reuse an expired capability, old source, closed deadline or previous checkpoint.
After an observation timeout inspect the original process/job; do not restart
solely because polling timed out. Grader faults are indeterminate and must not
be submitted as failures. Same-epoch duplicate task indices across miners score
zero; extra rollouts on one index earn no extra points.

## Current design and historical notes

See [COMMITTED_UNAUDITED_LEARNER.md](COMMITTED_UNAUDITED_LEARNER.md),
[CONTINUOUS_AUDIT_PREFILL.md](CONTINUOUS_AUDIT_PREFILL.md) and
[CONTINUOUS_STATISTICAL_REWARD_BRIDGE.md](CONTINUOUS_STATISTICAL_REWARD_BRIDGE.md).
The live public guide distinguishes deployed contracts from prospective trials.
[Earlier quickstart text](miner-quickstart-history-before-2026-10-06.md) is
archived for historical epochs and does not authorize the current mechanism.
