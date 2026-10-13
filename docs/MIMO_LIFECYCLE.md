# Prepared MiMo lifecycle

`ops/mimo_lifecycle.py` runs the new coordinator, its role queue, and an independent
audit scheduler. It does not modify or invoke the hourly weight writer. That
writer must keep its existing cursor, assessment history and last valid fallback.
The new assessment bridge supplies additional authenticated evidence to that
same writer; it does not create a second writer.

The example configuration at `ops/mimo_lifecycle.example.json` is deliberately
disabled and contains no credentials or usable release policy. Nothing in this
implementation installs a service or changes a discovery pointer by itself.

## Release admission

Create the exact source archive with `subnet.mimo_release.build`, including its
explicit task asset inventory. Include `ops/bootstrap_mimo_runtime.py`,
`ops/mimo_lifecycle.py`, `ops/mimo_lifecycle.example.json` and this document in
the builder's explicit `extra` argument: its default source list covers the Python
packages, not the operator entrypoints. Prepare the manifest with
`mimo_manifest.build`.
For a public release, set `payable: true` before building its policy, and supply
real authority-signed qualification receipts scoped to `policy_scope(...)`.
Required categories are runtime, cross-device sampling, mutation controls,
native grading, trainer update, model publication and held-out evaluation.
Receipts contain a hash of retained experimental evidence; unit test fixtures
are never release qualification evidence.

`mimo_release_policy.build_policy(..., enabled=False)` produces a review artifact.
An enabled signed policy pins the run, immutable checkpoint inventory, runtime
and template fingerprint, original task assets, source archive, sampler,
calibration, outcome projection and complete-group contract. The initial public
contract is eight trajectories with nonces 0–7, at least one completed correct
and one completed incorrect answer, and at most nine task groups per UID.

Public execution needs both an enabled qualified signed policy and an explicit
local `--allow-public-mimo` flag on the host and operator workers. Miner clients
authenticate the qualified signed policy through their pinned authority; they
do not possess the operator's local activation capability. Private policies
cannot authorize payable or public-named epochs.

## Reproducible worker installation

Use CPython 3.12 and `uv` on an NVIDIA H200 host with an appropriate CUDA 13 driver. Stage
the authenticated source release and the pinned runtime lock and rebuilt TOPLOC
wheel. The [prepared runtime inventory](https://affine.io/releases/mimo-runtime-20261013/ARTIFACTS.json)
provides permanent public, hash-addressed downloads. These files require no
bucket credentials and do not activate the new mining contract.
The native wheel must match the qualified torch installation. The older PyPI
TOPLOC binary failed its ABI import against this torch build.

The captured runtime lock and rebuilt wheel used in the private qualification
are published as `mimo-runtime-20261013`, marked **prepared, not active**, with
reviewed installer code at GitHub revision `96016dd0`. Download and verify them:

```sh
mkdir -p /operator/releases/runtime
MIMO_ARTIFACT_BASE=https://affine.io/releases/mimo-runtime-20261013
curl --fail --location --retry 3 --proto '=https' \
  "$MIMO_ARTIFACT_BASE/57442fdd51cf165c00ce4c531100aff3fc9f8a0a6ae43f4bf1868d5d0f912875/requirements.lock.txt" \
  --output /operator/releases/runtime/requirements.lock.txt
curl --fail --location --retry 3 --proto '=https' \
  "$MIMO_ARTIFACT_BASE/84c8db262fd302d21eb52c151ab039071948d0002f3848b09cea53c750934205/toploc-0.1.6-cp312-cp312-linux_x86_64.whl" \
  --output /operator/releases/runtime/toploc-0.1.6-cp312-cp312-linux_x86_64.whl
sha256sum --check <<'SHA256'
57442fdd51cf165c00ce4c531100aff3fc9f8a0a6ae43f4bf1868d5d0f912875  /operator/releases/runtime/requirements.lock.txt
84c8db262fd302d21eb52c151ab039071948d0002f3848b09cea53c750934205  /operator/releases/runtime/toploc-0.1.6-cp312-cp312-linux_x86_64.whl
SHA256
```

The lock contains exact package versions, including `torch==2.14.0` and
`transformers==5.14.1`. Do not substitute the separate `2.14.0+cu130` distribution:
its package fingerprint differs. Then install into a new directory:

```sh
python3 /operator/releases/mimo/ops/bootstrap_mimo_runtime.py \
  --destination /operator/runtimes/mimo-candidate \
  --source-root /operator/releases/mimo \
  --requirements /operator/releases/runtime/requirements.lock.txt \
  --requirements-sha256 57442fdd51cf165c00ce4c531100aff3fc9f8a0a6ae43f4bf1868d5d0f912875 \
  --toploc-wheel /operator/releases/runtime/toploc-0.1.6-cp312-cp312-linux_x86_64.whl \
  --toploc-wheel-sha256 84c8db262fd302d21eb52c151ab039071948d0002f3848b09cea53c750934205 \
  --python 3.12 --max-jobs 2
```

The destination must be new. This installs exact versions, checks dependencies,
runs the native TOPLOC round trip, and removes its own disposable package cache
after success. It does not download a model or activate any service. Inspect
`BOOTSTRAP.json`, `runtime-probe.json` and `install.log`. A successful installation
is not a public model qualification. Use this runtime's `venv/bin/python` for
the miner and operator workers; the existing signed-source launcher must point
at this interpreter and the authenticated source archive. Preserve each
miner's original key and private state directory.

## Operator configuration

Copy the example to an owner-only file outside the repository. Supply the
existing authority seed *path*, bucket credential *path*, signed release policy,
an isolated lifecycle state directory and an explicit queue database. The worker
roster maps verifier/trainer public keys to their allowed roles. The host runner
owns a loopback-only `CoordinatorServer` for this exact database and roster;
configure private tunnels to it. Do not point two independent queue APIs with
different rosters at this database.

The initial checkpoint object map contains each exact file name and its existing
R2 key. `source_files` is the exact `mimo_worker.source_inventory()` from the
prepared release; the host checks its running source against it. Runtime versions
pin torch, transformers and TOPLOC. The trainer role configuration file is an
absolute owner-only path on the trainer, with its publisher key and scoped R2
credentials. No private credential is placed in an opening or job report.
`source_bundle_key` names the exact approved R2 source archive. The host hashes
that bounded archive once per process, then renews its GET capability for every
new opening. Qualification and sampling bind the archive SHA and inventory,
not the expiring `url` or `read_url` aliases.

The validator needs only the checkpoint's authenticated tokenizer/config assets
in `tokenizer_directory`, not model weights. Each supplied file must match the
initial checkpoint inventory, and the template hash must match the release.
The public task snapshot is distributed inside the explicit source release.

Set `first_round` to the next global network round, not a reset of the existing
reward history. On a deliberate public cutover, set `previous_discovery_sha256`
to the exact existing discovery object's SHA256. Discovery uses conditional
PutObject and full readback; a changed predecessor is rejected. Unsupported
conditional operations fail closed. Private discovery is restricted to the
private run namespace.
The current object is a signed discovery descriptor containing `manifest_url`
and `manifest_sha256`, so existing direct-R2 signed-source launchers can follow
it to the immutable opening. CAS tracks the discovery object's SHA, separately
from the opening's SHA. A lost acknowledgement reads back the existing exact
descriptor instead of minting replacement URLs.

After qualifications and operator review, run the prepared configuration:

```sh
python ops/mimo_lifecycle.py --config /operator/private/mimo-lifecycle.json
```

Public mode additionally requires `--allow-public-mimo`. This document does not
authorize activating a live service. `--once` executes one orchestration pass.

## Epoch and recovery behavior

The host reads the existing chain activation path at a fixed block, snapshots
current UID-to-key mappings and publishes encrypted upload capabilities. It
captures committed compact documents throughout collection. After the deadline
it completes listing and capture, freezes the original population, grades each
eligible task natively, records exclusions and signs a whole-group training
dispatch. The existing role queue sends that dispatch to exactly one authorized
trainer. Sampling verification does not gate this dispatch.

The separate audit thread admits the signed frozen population and continuously
queues unpredictable audit selections, including historical populations. Its
failure does not change the current model, fabricate valid audit results, or
reset the reward writer. The assessment reader should use this scheduler's
directory and the same queue database, merged with preserved historical evidence.

An updated checkpoint is admitted only from the original authorized trainer's
signed queue report, exact publisher receipt and full streamed R2 file hashes.
The validator keeps a small readback receipt, not a local model cache. The
authority then signs the successor checkpoint authorization and advances its
local cursor. The next tick opens the next epoch. An empty epoch records
`no-update`, advances the epoch and randomness, and keeps the model and optimizer
step unchanged.

Lost upload/discovery/queue acknowledgements retry the same signed opening,
dispatch and job. Native grader or storage infrastructure errors preserve the
frozen state rather than penalizing miners or silently training partial data.
A terminal failed or expired training job returns `recovery-required` with its
original job ID and preserves the checkpoint and transaction journal. It is not
reported as healthy waiting and is not blindly reexecuted: recovery must establish
whether the optimizer already changed before supplying a new recovery action.
These states never stop the independent hourly best-assessment writer.
If the host saved an epoch plan but crashed before saving its signed opening,
and that unused window has expired, it records `abandoned-before-opening` and
plans a fresh epoch. This does not advance the optimizer or reward round and
does not rewrite an already published opening.
