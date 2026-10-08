# Start an Affine MATH miner

Read https://affine.io/llms.txt first. The latest signed OPEN manifest and its
approved source are authoritative; GitHub main includes default-off research.
Hourly current-assessment weight setting runs independently of compute epochs.
Historical nonpayable flags remain historical; follow the current signed policies.

Epoch 59 activates eight distinct rollouts per task batch:
four native-graded successes and four failures, at most three batches per UID
per epoch, and attempt nonces 0–999. Follow the signed OPEN manifest; historical
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

Read `authority` and `current_url` from https://affine.io/mining.json and set
`AFFINE_AUTHORITY` and `AFFINE_CURRENT_URL` locally. The expected authority is
3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd.
The key file contains your activated 32-byte Ed25519 seed as hexadecimal, mode600.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B -m subnet.source_bootstrap \
  --authority "$AFFINE_AUTHORITY" \
  --current-url "$AFFINE_CURRENT_URL" \
  --source-cache /private/affine-approved-source \
  --key /private/miner.seed \
  --state /private/affine-miner-cache \
  --env-id affine_math --max-batches 3 --search-budget 64 --once
```

Sixty-four is a local search budget, not the signed attempt ceiling. It may
need increasing for tasks where one outcome is rare; stop at the signed deadline.
v5 allows
1,000 attempts per task; old openings keep their original ceiling. For epoch 59, collect
four distinct successes and four distinct failures with eight distinct nonces.
The bootstrap verifies discovery, source and checkpoint hashes, then runs that
approved source. Signed code approval is not a sandbox. Miners need no permanent
R2 credentials; the manifest supplies encrypted private upload capabilities.

Watch discovery for each new OPEN epoch and run each only once. Never silently
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
