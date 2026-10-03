# Testing the new Affine miner

This is an experimental inference-verification training loop for subnet 120.
Miners download a signed checkpoint and environment challenge, find positive and
negative rollout batches, attach TOPLOC activation fingerprints and model
probability records, and upload privately through encrypted single-object R2
capabilities. After the deadline, the controller freezes submissions, publishes
an audit history, verifies/scopes duplicate environment indices, calculates
proposed weights and trains the next checkpoint. The next epoch waits for training.

## Current pilot status

Pilot epochs are permanently nonpayable: they do not set blockchain weights.
CPU multi-epoch trials and continuous GPU mining, verification, full-model
training and immutable checkpoint publication have run. Environment coverage
and hardware compatibility remain under active testing; use the runtime and
source bundle pinned by your approved challenge.
The next deployment opens admission to all live subnet 120 miner identities with
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

Once Arbos supplies the approved authority and a fresh discovery URL:

```bash
python -m subnet.cli \
  --gateway https://unused-gateway.invalid \
  --authority "$AFFINE_AUTHORITY" \
  --current-url "$AFFINE_CURRENT_URL" \
  --key /private/miner.seed \
  --state /private/affine-miner-cache
```

The key file contains the miner's 32-byte Ed25519 seed encoded as hexadecimal;
keep it private with mode 600. Never provide a coldkey, mnemonic, bucket credential
or wallet file to another person. Discovery is read-only; upload capabilities are
sealed to the miner and remain private. The manifest specifies model hashes,
environments, harness, K/L requirements, runtime, deadlines and upload limits.

Please test download/signature checks, rollout generation, cumulative uploads,
independent audits and checkpoint handover. Report the epoch ID, public hotkey,
error text with URLs/secrets removed, runtime and hardware. See
[LIVE_SUBNET.md](LIVE_SUBNET.md) and [ARCHITECTURE.md](ARCHITECTURE.md).
The dashboard is https://affine.io.

For the forthcoming single-MATH pilot, a clean checkout also needs the generated
task snapshot and exact approved source. Use the [signed-source bootstrap](MATH_MINER_BOOTSTRAP.md)
after the operator supplies an approved identity, runtime and live discovery URL.
The [MATH pilot plan](MATH_PILOT.md) describes the task split and pending launch
checks; preparation is not a public mining invitation or demonstrated improvement.
