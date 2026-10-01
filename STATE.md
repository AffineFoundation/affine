# Affine rewrite status

## Current verified state — October 1, 2026

GitHub main now contains the inference-verification rewrite. The deployed public
dashboard is affine.io; Arbos domains are unchanged. New-pipeline test epochs are
nonpayable and do not submit chain weights. Older operational entries below are
historical records, rather than instructions to re-enable their services.

The earlier small-held-out GPU series has five completed epochs with independently checked frozen submissions,
original-environment audits, 13 real optimizer updates, checkpoint publication,
and held-out evaluation records. Those updates consumed verified pairs from
Math, Verbatim, Reasoning Gym, When2Call, IFEval, and Oolong. These measurements
do not establish improvement across all environments.

The wider series has five fully checked epochs and fifteen full-model updates,
with original i3math, Trivia and PopQA tasksets added at versioned boundaries.
Six checkpoint objects per successor were independently streamed and hashed
(3,426,302,727 bytes each). The latest paired evaluation covers twelve
families and sixteen fixed held-out tasks per family: four metrics rose and
eight stayed flat after the PopQA update. Independent HTTPS checks matched
all 102 public evaluation records and the epoch/checkpoint/UID grids on affine.io.
The next fixed-reference optimizer source is running in a separate ongoing epoch.
These measurements
do not establish broad improvement. See [paired results](docs/WIDE_EVALUATION_RESULTS.md).

The controlled Tau2 tool-use positive/negative pair now passes independent
model verification and replay through the original environment grader. The
positive trajectory has six agent/user responses and reward 1; the negative has
eleven responses and reward 0. Both use the same task and checkpoint, and
auxiliary user tokens are excluded from training targets. Its agent-only
full-model optimizer completed one agent-only update. All six successor
checkpoint objects were independently hashed in R2, and a fresh six-role
rollout passed independent probability/TOPLOC verification and original
tool/grader replay. This is a controlled outcome-conditioned training example;
common epoch integration and held-out quality improvement remain unproven.

The controlled original Agent fixtures now run through the shared GPU epoch
pipeline. Eleven full epochs passed independent artifact, audit, score,
optimizer attribution, checkpoint-publication and held-out evidence checks.
One full-model update per epoch consumed its verified positive/negative pair; all six new
checkpoint objects were independently hashed (272,585,280 bytes). Two fixed
held-out tasks scored 0 before and after training, so no improvement is claimed.
The same miner has uploaded a new pair using the trained checkpoint. The
public actor/private grader images preserve original tools, mutable state and
grading; the full upstream Verifiers orchestrator remains unproven. See
[the shared native Agent scope](docs/NATIVE_AGENT_COMMON.md).

Original Spider SQL now has thirty-two pinned tasks, sixteen for mining and
sixteen disjoint held-out tasks. Public actors and private original graders
passed fresh remote controls. A controlled target-model candidate-policy
positive/negative pair passed independent full probability/TOPLOC checks and
native replay. Five common training epochs now passed independent evidence checks,
including resumed submission, auditing and training after an intervening empty
epoch. Each published six independently hashed checkpoint files and completed
sixteen fixed autoregressive held-out evaluations (0 before and 0 after).
The controller adopted bounded empty-search reporting at a completed epoch boundary.
These checks confirm the controlled loop, without
establishing held-out quality improvement. Calendar's curated positive/negative
six-turn model artifacts also pass signed model-audit binding and separate
original native replay. A new length-balanced Calendar pair additionally passes
seven model turns, including complete-context terminal proofs, and separate
fresh native admission with original rewards 1 and 0. See
[terminal control scope](docs/NATIVE_EOG_BALANCED_TERMINAL.md). These controls do not establish original sampling,
completed common Calendar epochs, or general quality gains.

Calendar's new common long-context controller completed its first full epoch.
Independent checks authenticated the signed manifests, frozen 175,247,173-byte
seven-turn positive/negative batch, full inference/native audit, normalized
proposed weight and one full-model agent-only update. All six successor
checkpoint objects were independently streamed and hashed (999,524,442 bytes).
Two fixed, disjoint original held-out tasks scored zero before and after, so
no quality gain is claimed. Its next epoch consumes the new checkpoint. Live
affine.io JSON and desktop/phone browser checks confirmed the Calendar chart,
exact evaluation records and 256-UID grid. The independent auditor accepts
these larger bundles only under the exact approved numerical/transport profile.
The newly published prospective worker rejects multi-step training until
persistent optimizer behavior is separately qualified. The active trial uses
one step and its frozen source remains unchanged. The public repo now includes
a guarded utility that reproduces its exact portable source variant in an
independent checkout; actual Git-archive reproduction passed without accessing
fixtures/models or starting services.

The public source-level matrix records 45 imports, 38 resets, 23 original-reward
and remote-proof milestones, 22 remote-replay milestones and 15 trained sources.
Pydantic has verified negative-only traces, with no qualifying K/L pair or training.
These flags describe specific demonstrated controls, rather than complete support
or broad task mastery across each environment family.

See [environment coverage](docs/environment-coverage.json) for the source-level
snapshot and [wide tasksets](docs/WIDE_TASKSETS.md) for measurement limitations.
Private operator evidence and the full completion audit remain under `state/`;
the full multi-environment goal is not complete.

## Historical implementation record

## Objective
Rewrite Affine for inference verification through sampling. The previously planned Wednesday teacher-model switch was publicly postponed.

## Workspaces
- New development workspace: /home/const/subnet120-rewrite
- Branch: rewrite/inference-verification
- Legacy running workspace: /home/const/subnet120
- Full legacy archive: /home/const/subnet120-archive/20260930T144912Z/repo
- Archive inventory and restore metadata: /home/const/subnet120-archive/20260930T144912Z/metadata

The full archive copy completed. All 1,396 tracked entries matched recorded checksums, and archived Git connectivity passed. Runtime logs/state were copied live and are not an atomic database snapshot.
Legacy source removals are staged only on the rewrite branch. No rewrite commit or remote push has been made.

## Established access
GitHub PAT authenticates as unarbos, with access to AffineFoundation/affine.
Arbos Discord bot can post in Bittensor SN120 channel 1381987595881414656.
Validator wallet: /home/const/.bittensor/wallets/default/hotkeys/default (Finney netuid 120). Private keys remain outside this checkout.
1Password vault: Arbos. Credentials are not copied into this workspace.

## Existing operations
Public chat was persistently disabled and its pod removed.
The legacy validator and support services remain active. The operator deferred the shutdown decision and requested a standalone hourly owner-weight burn script; that implementation is underway in a separate sub-agent.
No GPU pod is terminated by workspace preparation.

## Next design decisions
1. Define the work miners perform and the inference-verification protocol.
2. Define sampling, scoring, fraud handling, and incentive rules.
3. Specify validator/miner interfaces and compute/storage requirements.
4. Plan migration, rollout, and restoration of the previous system if needed.

## Hourly owner-weight heartbeat
Installed independent owner-only SN120 weight job: `ops/hourly_burn.py`; operator reference `ops/HOURLY_BURN.md`. Runtime is `~/.local/share/affine-burn/` with dedicated Bittensor 11.0.2 venv; user-systemd timer `affine-hourly-burn.timer` is enabled and active at each UTC hour, with user lingering enabled. Owner resolved live to hotkey `5HmYnmUYT6qe3yFMg1Ad8WLLqDvnwtjYakXBpDvoRW1Qqzb8`, UID 0. Commit/reveal is enabled and automatically handled by SDK. The first manual invocation was deferred because the live validator had recently set weights (100-block limit); no extrinsic was sent by the new job. Legacy services remain untouched. The old validator can overwrite these weights or keep the hourly job rate-limited until the operator decides its transition.

## Verification experiment (2026-09-30)

Operator delegated an end-to-end GPU prototype and explicitly requested retaining its machine for follow-up experiments. New experimental Lium pod `affine-proof-experiment` / HUID `golden-comet-16`, ID `8c416640-e9dd-453a-b78f-ce1d9a866df8`, single RTX3090 24GB, $0.22/hour, SSH `root@90.95.12.246` port20062; created15:35:12UTC, no removal schedule. At15:52UTC Lium reported $0.06 spent. This is separate from all production pods.

`prototype/pipeline.py` ran pinned SmolLM2-1.7B-Instruct through upstream Prime Intellect Mastermind, generated a genuine3-turn trajectory, full-vocabulary log-probabilities and12 TOPLOC fingerprints, then independently verified it in a fresh process. Positive curated control accepted as intended. Final report:2 accepted controls and14 rejected resealed adversarial artifacts, including actual changed-layer computation and isolated changed-weight fingerprints. Full warm verification about0.47seconds; same GPU/backend/bfloat16 only. Trusted weights policy is outside miner ZIPs. See `prototype/README.md`, `prototype/pod.json`, and local/remote `prototype/artifacts/report.json`; full weights+trajectory are in `complete-rollout.tar`. Pod source is `/root/affine-proof`; Python `/root/proof-venv/bin/python`. TOPLOC0.1.6 had to be rebuilt for torch2.12.0 ABI. Production environment remains unchanged. Next: repeated/cross-hardware trials and random partial-audit design.

## Equal registration transition — 2026-09-30
User replaced owner-only burn with 1/N for model registrations activated from now onward. Burn timer disabled; affine-transition-weights.timer active every minute. Cutoff Finney block 9181759, 15:54:34 UTC. Verified new uploaded R2 models only; no current eligible miners at deployment. Original validator restarted online with persistent marker payout guard, intake/evaluation retained. Submission window ends October 2 15:54:34 UTC; no automatic old payout restoration. See ops/EQUAL_REGISTRATION_WEIGHTS.md.

Experiment artifacts are fully retrieved locally. All16 attack/control ZIP SHA256 values match the report; the3,438,131,200-byte full-state bundle matches its remote SHA256 `430d9d27540a6eb77cb9daafdf4cbb6d23a5f123a9522c73d614f673ea3ecec5`. Billing ledger through15:55:01UTC:1,189.86seconds, $0.0727136884. The experimental pod remains running at $0.22/hour.

## Extended inference checks (2026-09-30)

Local isolated TOPLOC0.1.6 extension rebuilt for torch2.14.0+cu130 using privately extracted Python3.12 development headers; `.venv/bin/python` now imports and verifies native proofs. Transformers5.14 BatchEncoding chat-template return handled in prototype context helper. Experimental GPU was externally removed: Lium ledger `user_initiated`17:20:20UTC, final$0.382979/6,266.93seconds; this agent did not terminate or re-rent it. Full approved1.7B state is preserved locally.

CPU-vs-originalGPU **cross-runtime** test (hardware and software versions both differ) rejects honest GPU proofs at exact settings. Worst honest exponent mismatches3/128, mean mantissa error1.76, full logprob delta0.62977. One post-hoc tolerant point admits altered-weight computations with layer scales1.003/1.01, despite rejecting the original four numeric attacks. These tolerances are diagnostics only, not deployed.55frozen-artifact random audit configurations/1.1million sampled challenges match hypergeometric detection predictions; cached reference computations mean this measures detection rather than savings. See prototype/EXTENDED_RESULTS.md and prototype/artifacts/extended JSON reports. Production services untouched; no additional rental.

## Complete synchronous pipeline — 2026-09-30

Implemented `subnet/` controller, private cumulative R2 upload gateway, signed epoch
manifests, Ed25519 sealed capabilities, miner search/upload client, bounded ZIP
transport, independent full-audit verifier, uniqueness scoring, separate trainer
subprocess, checkpoint publication/next-epoch barrier, fresh-chain registration
adapter and hourly SDK payout worker. See IMPLEMENTATION_PLAN.md and
[architecture](docs/ARCHITECTURE.md).

Actual dedicated R2 mock completed epoch `mock-1790789692`: three local miner keys,
real135M target-model sampled guesses, full-vocabulary probabilities/TOPLOC proofs,
independent verifier processes and environment replay. Two miners collided on
index0 and each had one unique valid index, producing points1/1/0 and weights.5/.5/0.
Third miner's corrupted reward rejected. Private access403, wrong-key decryption
failed, late overwrite403, frozen public SHA verified. Real preference training
changed weights; checkpoint `97f3406807b61835a497eeba9ff82b58244b722632065d2a8786b230f0b1e3d8`
was uploaded and its next-model rollout verified. No blockchain transactions.
Private evidence `state/e2e/report.json`; later separate trainer-process proof in
`state/e2e/separate-trainer-report.json`. Two protocol and nine chain-adapter checks
pass. This is conservative full-audit smoke, not a finalized research objective.

Live read-only chain inspection found212 current signed Ed25519 registrations at
block9182279; actual participation in the NEW pipeline not yet demonstrated.
Payout cutover and reachable HTTPS controller deployment remain to be verified.
Original transition writer still active as of this record; no new chain extrinsic.
Cross-backend numerical compatibility is not proven and tolerances remain strict.

Final current-source mock also completed `mock-1790790431` with separate trainer
subprocess and invalid miner intentionally colliding with valid index1. Scores stayed
1/1/0, verifying invalid data cannot cancel valid competitor. Evidence
`state/e2e-final/report.json` success=true. Cached finalization was separately checked
idempotent, retaining the exact finalized timestamp/report. Service units verify.

Live controller and dedicated temporary Cloudflare quick tunnel were started,
without altering old production ingress. Independently fetched/verified signed
manifest over HTTPS: `live-1790790840-0`,212 registered miner mailboxes,8000 indices,
deadline2026-09-30T18:04:00UTC. Temporary gateway
https://takes-binding-pastor-biblical.trycloudflare.com; authority public key
`cebf616ae4cdd1a78a4441040046a1714522dfb6d4e8ecedbdd3974f66a09646`.
Local evidence `state/live/readiness.json`. New controller is collecting, not proof
of actual miners or accepted live weights. Transition payout timer remains active
until root cutover; no new mechanism extrinsic. Quick-tunnel hostname changes on
restart: replace it with a durable named ingress before claiming durable operation.
Full-audit CPU tiny Mastermind/135M checkpoint are explicit smoke settings.

## Registered remote trial steering

Operator now requests remote owned-registered-miner trial **without chain payouts**.
No payout cutover occurred; new payout timer/service inactive. Added permanent
nonpayable report flag and payout filtering of `nonpayable-`, `test-`, `mock-` epochs;
ten chain checks (including nonpayable exclusion) plus two protocol checks pass.
`subnet.registered_test` isolates test uploads/epochs and never writes payout ledger.
Remote miner may receive only an expiring scoped capability after local mailbox
decryption, so the operator wallet need not be transferred.

Rented and retained separate RTX3090 pod `affine-registered-miner-test`,
`816e4656-05f5-4b11-afee-cd823068ef17` / `gentle-fox-99`, $0.22/hour,
SSH root@90.95.12.246:20062; new container at prior experiment's endpoint, with a
separate explicitly accepted host-key file. No scheduled removal. Exact CPU
runtime installation is underway in `/root/miner-venv`; global image torch2.12 pip
constraints were overridden for isolated torch2.14 install.

Identity discovery is blocked:17 original miner wallets and validator are sr25519;
no Ed25519 seed/public matches in backups. Relevant authorized vault mining and
validator mnemonic records likewise correspond to sr25519. Their Ed25519 derived
alternatives have no SN120 UID at fresh block9182516. See public-only evidence
`state/registered-test-discovery.json`. Transient private vault copies were deleted.
An already-registered owned Ed25519 private-key source is needed; no replacement
identity or new on-chain registration is fabricated. Existing transition writer
may independently transact; this trial has submitted no chain extrinsic.

## Owned registered remote trial (2026-09-30)

One explicitly authorized paid Ed25519 miner registration succeeded on Finney SN120 at block 9182681, UID131, hotkey `5E68nqmVj1gSjoJHbusG2o4SiJ1dq17M7QG5PVzFKK2ic49u`. Actual funded coldkey balance decrease including burn+fee:1.061166556TAO; remaining95.669061463TAO. Registration evidence `state/registered-test-registration/receipt.json`; activation commitment confirmed and visible at block9182704. Coldkey and Ed25519 private key remain on operator host.

Retained Lium miner pod `816e4656-05f5-4b11-afee-cd823068ef17` (`gentle-fox-99`, affine-registered-miner-test), RTX3090, $0.22/hour, SSH root@90.95.12.246:20062; only expiring encrypted-mailbox-derived upload capability delegated to host, no coldkey exported. Pinned CPU inference runtime installation in progress. Isolated nonpayable epoch job is `state/registered-test/job.json`, temporary HTTPS ingress `https://the-organization-initial-latex.trycloudflare.com` forwarding8792. Port8790 was already occupied by production uvicorn and was left untouched; discarded first trial tunnel stopped.

New payout writer and cutover remain DISABLED. Trial epoch prefix `nonpayable-` and `payable:false` are excluded independently from hourly points. No payout ledger inputs are written by trial runner; registration+activation were setup transactions, no chain weights submitted by this experiment. Existing unrelated transition writer remains active.

### Registered trial progress19:36UTC

Aligned CPU execution profile (MKL_CBWR=COMPATIBLE, ATEN_CPU_CAPABILITY=default, ONEDNN_MAX_CPU_ISA=SSE41) allowed the honest remote batch to pass unchanged strict full-logprob/TOPLOC checks. Independent audit accepted index2; frozen SHA d8cf3fb11d00fc537efdc02a3376509ca6fc67c2a74a45b76257c32356d12168. Real trainer step completed with loss0.6931471824645996 and changed weights. Initial tokenizer publication omitted Transformers5 chat_template.jinja; model_files now includes it. Repaired immutable checkpoint2b80bfabb0b54a409d8fb4df832112d208773d40391a76e5e2475d3306f31166 is public in nonpayable-registered-1790795876-next-complete.

Unexpected external interruption: Lium pod816e4656-05f5-4b11-afee-cd823068ef17 was deleted19:34:48→19:35:27UTC, authoritative reasonuser_initiated. This agent did not issue deletion/reboot/TTL. Saved describe evidence state/registered-miner-pod-current.json. Final next-round remote sampling/verification has NOT completed. Health source state/registered-test-compatible/health.json truthfully blocked_remote_pod_deleted; prior accepted batch and training remain valid. Root was notified to reconcile possible user cancellation before any replacement rental. No weights/cutover were executed.

## Public dashboard — September 30
User mockup implemented and deployed to affine.io: monochrome Affine wordmark, hero batches-per-epoch chart,256-cell UID grid. SQLite public projection refreshed every15seconds by affine-network-dashboard.service; exported network-data.json uses only allowlisted public metrics. Pilot/nonpayable mode separate from live; UID131 recorded batch1. Desktop/mobile and public HTTPS checks passed. No Arbos/DNS edits. Dashboard README has backup/rollback paths.

### Registered remote trial COMPLETE (20:07 UTC)

The root cause of the original rental removal was the existing `affine-pod-reaper`: the rental's required ownership entry was omitted, so it removed the pod after 90 minutes. Replacement pod `0a67adc9-b484-4172-96c1-073efec7d8be` (`zesty-comet-b2`, `affine-registered-miner-retained`) is RUNNING, RTX3090 at $0.22/hour, SSH `root@90.95.12.246:20059`. It is registered explicitly as `manual:user-retained-verification` with expected_hours=0 (indefinite); global production cleanup was not disabled. Exact known tested runtime was copied as a portable package bundle. No coldkey/private hotkey was exported.

The remote replacement downloaded the complete trained checkpoint, generated and uploaded another positive/negative batch, and passed the independent full audit at environment index 1. Frozen artifact SHA `c1f3a9aadb97438fd235c15cd64c1a57cc7a41780b5287ba24ad83f611ed50e6`. Final epoch `nonpayable-registered-1790795876-next-retained`. The initial accepted epoch index 2, genuine separate-process training step, publication of changed weights, and fresh next-epoch remote verification are all complete. First failed unconstrained CPU audit and incomplete tokenizer-packaging attempt remain historical evidence; thresholds were not loosened.

Final report and health `state/registered-test-compatible/report.json` / `health.json` have complete=true / status=complete / next_remote_rollout_verified=true. UID131 Ed25519 hotkey ownership rechecked at block9183026. All test scores remain nonpayable, future payout points empty, no chain weight submission or writer cutover occurred. Existing unrelated transition writer remains active; new payout worker inactive.

13 protocol/chain tests pass; four trained-checkpoint attack controls (forged reward, forged feedback, forged full log-probs, substituted TOPLOC proof) reject with expected reasons. `trained-checkpoint-tampering-controls.json` records them. Final public experiment, health, chain-activity and tampering-controls reports are signed under first accepted epoch at the retained temporary HTTPS gateway. Gateway service retains audit access; remote pod stays running indefinitely as user requested.

## Latest nonpayable multi-environment milestone — 2026-09-30 21:52 UTC

Three actual epochs completed on the retained remote registered UID131 miner:
Prime Verbatim, Prime Mastermind, and original reasoning-gym count_bits. Each
provided an accepted positive/negative batch, full independent target-model and
environment replay audit, local proposed weights, one genuine training step and
publication of the next immutable checkpoint. Final checkpoint:
`d53bb84e8eddd39b4168800e6744359ae4b399a13e5c8c6676292b1c12b365ef`.
All reports are permanently nonpayable; this experiment made no weight extrinsic.
The older transition service is separate and may still operate independently.

Current evidence: `state/multi-environment/report.json`, `progress.json`,
`status.json`, and cumulative `state/evaluations/*.json` (operator-owned metrics
consumed by affine.io). Same fixed-task before/after means are Verbatim1→1,
Mastermind1→1, count_bits0→0. Those are curated control policies; no training
improvement or autonomous-solving claim follows. A separate fixed original-task
suite is continuously evaluating baseline and newly trained checkpoints, with
honest unsuccessful autoregressive samples retained as valid evaluation evidence.

The active nonpayable service is `affine-multi-environment.service`; stop/start
instructions and exact numerical profile are in `docs/CONTINUOUS_EXPERIMENT.md`.
The immutable source implementation for these three epochs is retained under
`state/source-bundles/`, with an authority-signed public descriptor. Runtime v2
explicitly bounds TOPLOC native bit-extraction threads to avoid its upstream
hardware-concurrency override while retaining strict numerical comparisons.

An additional real 1.7B GPU original-Math inference/replay pilot accepted its honest
trace and rejected five tampering controls, reward0, with exact same-GPU backend
weights reload. Evidence `state/multi-environment/gpu-pilot/report.json`. That is
larger-model proof evidence, not training or cross-hardware compatibility.
