# Testing the new Affine miner

This is an experimental inference-verification training loop for subnet 120.
Miners download a signed checkpoint and environment challenge, find positive and
negative rollout batches, attach TOPLOC activation fingerprints and model
probability records, and upload privately through encrypted single-object R2
capabilities. After the deadline, the controller freezes submissions, publishes
an audit history, verifies/scopes duplicate environment indices, calculates
proposed weights and trains the next checkpoint. The next epoch waits for training.

## Forced sampling update — October 4, 2026

When the signed manifest contains `forced-inverse-cdf-replay-v1`, use the newly
admitted source and prescribed sampler. The epoch fixes randomness, checkpoint,
task, attempt, turn and token position. Attempts range from zero to the signed
limit minus one (currently 128 attempts). The source chooses the corresponding
draws; an arbitrary seed or synthesized answer with valid TOPLOC does not qualify.
Audited outputs must exactly replay, including stopping. The qualified launch
profile is H200/SM90 with the exact packages and numerical settings in the
manifest; older RTX 3090 qualification does not establish compatibility.

Read https://affine.io/mining.json and the signed manifest for actual activation
and upload availability. The contract activated from epoch nine on October 4;
earlier epochs keep their original rules. The current pilot quota is three
batches per identity per epoch, each
containing one successful and one unsuccessful trajectory for a distinct task.
Only fully audited valid batches earn reward points or enter training. See
[FORCED_SAMPLING_QUALIFICATION.md](FORCED_SAMPLING_QUALIFICATION.md) for controls,
costs and limitations; they are not a completed live epoch or a learning gain.

## Historical launch preparation — October 3, 2026

The operator authorized prospective live MATH rewards. At that point preparation
was under way and weight setting was not live. Follow https://affine.io/llms.txt and
https://affine.io/mining.json for actual opening and payout status. Use an agent
to manage mining and watch GitHub main for continual updates. Signed epoch source
pins are authoritative; rerun the signed-source bootstrap when the source changes.
Historical pilot epochs remain nonpayable. The new bounded sampler and reward
parameters are documented in [AUDIT_SAMPLING.md](AUDIT_SAMPLING.md).

## Historical pilot status

Pilot epochs are permanently nonpayable: they do not set blockchain weights.
CPU multi-epoch trials and continuous GPU mining, verification, full-model
training and immutable checkpoint publication have run. Environment coverage
and hardware compatibility remain under active testing; use the runtime and
source bundle pinned by the signed challenge.
The live v9 deployment opens admission to all live subnet 120 miner identities with
a valid signed Ed25519 Affine activation. Its config uses
`registration_policy: all_activated_subnet` and omits `registration_allowlist`;
no individual operator approval is needed. Each new epoch takes a fresh chain
snapshot, so later activations enter the next epoch. Existing signed epochs keep
their original participants. See STATE.md for actual deployment progress and use
the signed discovery URL and runtime profile for the open epoch before renting
compute. Performance improvement across all environments is not established.

## Prepare a miner

Clone https://github.com/AffineFoundation/affine and use Python 3.12 or newer.
Create a virtual environment and install this repository with `pip install -e .`.
Exact model, execution profile, environment dependencies and source bytes must
match the signed challenge; a CPU challenge does not authorize a GPU runtime.

For an existing subnet-registered Ed25519 hotkey:

```bash
python -m subnet.register --wallet YOUR_WALLET --hotkey YOUR_ED25519_HOTKEY
```

This previews activation. Add `--execute` only when ready to publish the signed
activation commitment. It does not purchase a new subnet UID. Subnet membership
and a valid signed activation are both required; open admission removes the
operator allowlist, not identity authentication.

Read `authority` and `current_url` from https://affine.io/mining.json and set
`AFFINE_AUTHORITY` and `AFFINE_CURRENT_URL` locally:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B -m subnet.source_bootstrap \
  --authority "$AFFINE_AUTHORITY" \
  --current-url "$AFFINE_CURRENT_URL" \
  --source-cache /private/affine-approved-source \
  --key /private/miner.seed \
  --state /private/affine-miner-cache \
  --env-id affine_math --max-batches 3 --search-budget 8 --once
```

The key file contains the miner's 32-byte Ed25519 seed encoded as hexadecimal;
keep it private with mode 600. Never provide a coldkey, mnemonic, bucket credential
or wallet file to another person. Discovery is read-only; upload capabilities are
sealed to the miner and remain private. The manifest specifies model hashes,
environments, harness, K/L requirements, runtime, deadlines and upload limits.
The local search budget limits attempts on each task before trying another.
Eight is a starting recommendation, not a change to the signed 128-attempt
ceiling or sampler. Spending every attempt on one task that always succeeds or
always fails can exhaust the epoch without producing a pair. Supported budgets
range from one through 128; the miner must still follow the same public draws
and submit a complete verified success/failure pair.
Watch for each fresh epoch and repeat the bootstrap once with its current
source/checkpoint. Do not restart an existing attempt merely because observation
timed out. The bootstrap downloads the approved task asset and source, verifies
their hashes, and launches that source in an isolated interpreter.

Please test download/signature checks, rollout generation, cumulative uploads,
independent audits and checkpoint handover. Report the epoch ID, public hotkey,
error text with URLs/secrets removed, runtime and hardware. See
[LIVE_SUBNET.md](LIVE_SUBNET.md) and [ARCHITECTURE.md](ARCHITECTURE.md).
The dashboard is https://affine.io.

For the forthcoming single-MATH pilot, a clean checkout also needs the generated
task snapshot and exact approved source. Use the [signed-source bootstrap](MATH_MINER_BOOTSTRAP.md)
with your registered, activated identity and the published runtime and live discovery URL.
The [MATH pilot plan](MATH_PILOT.md) describes the task split and pending launch
checks; preparation is not a public mining invitation or demonstrated improvement.


Prospective hourly transport: see [SMALL_COMMITMENT_HOURLY_CONTRACT.md](SMALL_COMMITMENT_HOURLY_CONTRACT.md) for the versioned small-commitment, selected-only audit contract. It applies only when the signed epoch manifest explicitly activates that policy; existing epochs retain their original contract.
