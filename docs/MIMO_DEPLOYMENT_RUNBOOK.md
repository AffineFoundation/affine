# MiMo deployment and rollback

This runbook prepares a controlled transition. The new public release is not
activated by installing code or uploading artifacts. Existing learning and the
sole hourly reward writer continue while qualification runs.

## Existing live boundaries

The validator host is Arbos.life, the current development machine. These are
user systemd units, so operational commands use `systemctl --user`:

| Responsibility | Existing unit | Transition rule |
| --- | --- | --- |
| Current learner | `affine-ordinary-v3-native-eligibility-durable-learner-v1.service` | Finish and archive the current epoch before replacing epoch ownership. |
| Current verifier API | `affine-independent-nine-source-durable-api-v1.service` | Keep historical work and reports available; new MiMo queue is isolated. |
| Current continuous auditor | `affine-continuous-nine-source-durable-auditor-v1.service` | Drain old jobs; do not relabel their source or numerical policy. |
| Hourly weight writer | `affine-live-reward-writer.service` and `.timer` | Preserve the same timer, transaction cursor, singleton, state and fallback. |
| Public mining status | `affine-mining-discovery.service` | Replace this one service's entry, never run two exporters for the same file. |
| Dashboard data | `affine-network-dashboard.service` | Introduce new run data through a data adapter; preserve page layout and incentive history. |

The old `affine-live-math-controller-v1.service` is archived and masked. It is
not a rollback target. The current learner and writer are authenticated signed
compositions; their live policies and commands are the recovery references.

The mining observer currently runs from
`/home/const/subnet120-rewrite/ops/publish_live_public_discovery.py` and writes
`/home/const/subnet120/affine/website/mining.json`. It now understands the active
verifier-only wrapper while retaining the original generation/config bindings.
An authentic `between_epochs` status during training must not be mistaken for
an open collection window.

The active R2 discovery key, derived from the authenticated learner's
`direct-discovery.json`, is
`public/streams/nonpayable-live-reward-math-v1-/current.json`.
The top-level `public/current.json` is an old mock under a different authority;
it is not the live handoff target. Re-read and hash the active stream at the
actual boundary, because its current epoch changes continuously.

## Freeze and stage

1. Build the final release from `/home/const/subnet120-mimo-integration`, which
   preserves current GitHub main. Do not deploy the old rewrite checkout as a
   replacement for main. Include the exact source closure, licensed pinned
   MiMo reference, explicit MATH asset and all required operator entrypoints.
   The additional observer files are `ops/publish_mimo_public_discovery.py`,
   `ops/mimo_mining_observer.example.json` and
   `ops/mimo_mining_observer.service.conf.example`. Include the initial asset
   staging tool and this runbook too. Freeze only after active code changes end.
2. Upload the immutable source archive and exact runtime dependency artifacts
   to content-addressed private keys; fully GET and SHA256-check every artifact.
   Keep GET capabilities separate from immutable scientific/source bindings.
   Never expose credentials. Rebuild the runtime with
   `ops/bootstrap_mimo_runtime.py` as documented in `MIMO_LIFECYCLE.md`.
   Build the manifest's `source_bundle` with `mimo_release.source_bundle(record)`:
   it uses `archive_sha256` and `archive_bytes`. The separate `source_sha256`
   hashes the uncompressed file inventory and cannot pass the miner's archive
   download check. Worker file hashes remain independently pinned.
3. Stage the **original** MiMo checkpoint from the authenticated `FETCHED.json`
   inventory using `ops/mimo_stage_initial_checkpoint.py`. It uploads only model
   files with conditional multipart writes and full streamed readback. Its
   receipt is explicitly an asset receipt, not a training result or public
   qualification. The expected initial checkpoint is
   `87de8304e9d33dde39b81a41fdbbf306426ea6fd993b7d85332e6a48e8a4bbae`,
   18,849,863,399 bytes, from model revision
   `2367e865d009c13ac81713a2878291d33ab28177`.
4. Retain the existing authority; generate/assign worker keys with explicit role
   rosters. The host needs tokenizer/config assets only. New worker source,
   runtime, native TOPLOC binary, model and template must match the release.
   Trainer optimizer state remains local, with one trainer and its own lease.
5. Prepare an owner-only disabled lifecycle config in a new state directory,
   based on `ops/mimo_lifecycle.example.json`. Fill actual worker/publication
   keys, source inventory, model object keys and runtime versions. Keep the
   new loopback queue at `127.0.0.1:19183` and configure private tunnels before
   the first public opening. Run independent audit and evaluation schedulers.

The first operational configuration targets 1,800 seconds of collection and
32 distinct eligible task groups per optimizer update, pending real long-output
training and model-publication timing. The implementation accepts up to 512;
this is an explicit training budget, not a change to the miner contract of
eight prescribed trajectories and nine groups per UID. Measure the complete
epoch before claiming an hourly service level.

## Readiness evidence

The authority may enable the public release only with real scoped receipts for
runtime, cross-device sampling, mutation controls, native grading, training
update, model publication and held-out evaluation. CPU tests and diagnostic
receipts cannot stand in for the real model. A fabricated or incomplete group
must not become training data, but sampling audits remain independent of the
training dispatch.

Before activation, run the actual external miner launcher against an isolated
signed descriptor and the staged source/runtime/model GET paths. Show complete
nonce groups, compact upload, capture, native grading, trainer update, immutable
publication, audit queue/report admission, evaluator result and a successor
opening. Verify downloaded artifacts from a machine with no pre-existing model
cache. Record unfinished/EOS outcomes and full model publication wall time.

## Reward writer continuity

Preserve the sole deployed writer even if MiMo qualification, learning,
discovery or audits fail. Its existing assessment history, six-hour EMA,
hourly cutoff cache, transaction cursor and current-registration remapping
remain authoritative. Do not reset weights, burn, or install another writer.

New MiMo audit evidence is composed into the original evidence-loader callback
using `mimo_assessment_bridge.EvidenceSource` and `compose_loader`. The deployed
writer's pinned `ops.current_assessment_evidence` supplies its existing
`snapshot`, `current_estimates` and `observations` callbacks. Load the new
bridge under an isolated package namespace so it cannot shadow the old
`subnet` imports. A new signed wrapper must retain the original writer's
parent policy, cutover, anchor, lock and state paths. Newly issued verifier keys
must be included in an explicitly signed roster policy; a queue roster alone
does not authorize their reward evidence.

Set the MiMo `first_round` to the last committed network round plus one.
Do not restart numbering at zero. The first MiMo evidence can legitimately
appear at the next hourly cutoff because the current hour's authenticated
assessment may already be cached. Preserve the last valid assessment while
new evidence is unavailable. Retain every old population and signed report.

## Controlled cutover

1. Snapshot read-only service commands, signed policy hashes, last completed
   epoch/checkpoint/optimizer identity, current R2 discovery SHA and writer
   cursor/assessment status. Store owner-only references; never copy secrets
   into a public runbook. Record the sole observer's unit and existing drop-ins.
2. Wait for an authenticated safe epoch boundary. Preserve the old learner's
   completed state and optimizer; stop only that learner's epoch ownership
   before enabling the new one. Do not stop the hourly writer or erase history.
3. Enable the qualified public lifecycle configuration and start its one
   prepared host service. Workers use the qualified release and an explicit
   local `--allow-public-mimo` opt-in. The public discovery key update uses the
   exact previous object SHA and conditional write/readback, never blind PUT.
4. Confirm the new host process owns the exact config/source and has produced
   `CURRENT.json`, an authenticated opening and its original `DISCOVERY.json`
   receipt. An empty epoch advances without a fake training step. A terminal
   failed training job must be `recovery-required`, not silently restarted.
5. Prepare an owner-only observer config from
   `ops/mimo_mining_observer.example.json`. Pin the enabled lifecycle config
   SHA, qualified source SHA, lifecycle entry SHA and authority. The observer
   rejects private qualification releases and checks the physical process.
6. Hand off the **existing** observer unit by installing the reviewed drop-in
   as `~/.config/systemd/user/affine-mining-discovery.service.d/90-mimo.conf`.
   Substitute the authenticated release and runtime paths in the example.
   This is the only exporter restarted:

   ```sh
   systemctl --user daemon-reload
   systemctl --user restart affine-mining-discovery.service
   systemctl --user show affine-mining-discovery.service -p MainPID -p ActiveState
   ```

   `ops/publish_mimo_public_discovery.py` uses the original signed discovery
   document retained in `DISCOVERY.json`. It needs no R2 credential or authority
   seed. Verify `mining.json` shows the real open interval, nonce range 0–7,
   eight trajectories, minimum one success and completed failure, cap nine,
   correct source/checkpoint and unexpired GET descriptor. Do not invent an
   accepting state while the host is training or unavailable.
7. Change the dashboard's data source to the new authenticated run/evaluation
   adapter, preserving its appearance and earlier runs. Publish the compatible
   main code and current miner documentation, then announce. Observe actual
   external submissions and a full successor epoch, not only the operator miner.

## Rollback and failure recovery

An operational rollback never rewrites an issued manifest or clears the reward
history. Close further openings on the new lifecycle; preserve active jobs,
transaction journals and signed reports. Establish whether the trainer already
stepped before retrying anything. Keep the existing best-assessment writer
running throughout.

If no MiMo training step has committed, the archived old learner can resume from
its original authenticated checkpoint/optimizer and next unused epoch, using
its exact saved signed composition. After a committed MiMo step, retain it as
an independent run; any model rollback must be an explicit new model lineage,
not a claim that the step never happened. Public discovery rollback is another
authenticated conditional transition with a fresh valid signed opening.

To restore only the observer, remove **only** the exact reviewed `90-mimo.conf`
drop-in, verify that all original unit/drop-ins still have their saved hashes,
then daemon-reload and restart the same `affine-mining-discovery.service`.
Do not remove unrelated unit overrides or restart an archived controller.
The old observer must point to an actually running old learner before it can
advertise mining. Restore dashboard data independently; never change reward
writer state as part of a website rollback.
