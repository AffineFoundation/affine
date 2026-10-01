# Affine epoch subnet: implementation and rollout

## User-authorized outcome

Build the entire subnet in this repository: miner, registered identities, epoch
controller/validator, storage access, offline verifier, scoring, trainer, history,
and live Finney SN120 weight submission. First demonstrate an end-to-end mock with
no blockchain transactions, real inference/proofs and a real training update.
Then integrate and operate the complete pipeline on the network. The user
explicitly requested delegation and authorized use of available Cloudflare/vault
credentials and the validator wallet. Proceed autonomously on authorized work;
never claim live deployment or accepted weights without evidence.

## Agreed protocol

1. Synchronous epochs. Publish immutable checkpoint weights and a signed epoch
   manifest containing model/tokenizer/config hashes, pinned environment version,
   indices, start/end times, K positive and L negative samples required per batch,
   audit policy/version and encrypted upload capabilities for registered miners.
2. Miners register using Ed25519 identities. Reuse the previous registration and
   sealed-box mailbox design after confirming ownership against chain registration.
   Ed25519 signing keys require their supported Curve25519 conversion for encryption.
   Do not silently treat arbitrary sr25519 keys as encryption keys.
3. One private cumulative batch file per miner per epoch, repeatedly overwritten
   while the epoch is open. A batch targets one environment index, contains distinct
   positive/negative trajectories with tokens, observations, outputs, probabilities,
   scores and TOPLOC fingerprints. No miner-operated bucket is required.
4. At the deadline block all further writes, wait for/resolve in-flight writes,
   freeze final objects with hashes, and publish immutable copies for public audits.
   Cloudflare R2 does not implement arbitrary folder ACL flips; keep private upload
   objects and expose frozen public objects through a narrowly scoped gateway.
   Expiring a URL alone is insufficient to prevent an in-flight overwrite race.
5. Offline verification workers audit frozen artifacts. Cheap structure/checkpoint/
   environment/count/duplicate checks run on every batch. Random expensive checks
   recompute target-model activations/probabilities and replay environments. Publish
   signed immutable reports referencing exact frozen hashes and sample selection.
6. Uniqueness scoring supersedes first-arrival scoring: one point for a qualifying
   environment index submitted by exactly one miner in that epoch. If multiple
   miners submit qualifying batches for the same index, all receive zero for that
   index. Invalid batches must not cancel a valid batch. Deduplicate within each
   miner. Scope indices by epoch/checkpoint. Multi-identity collision attacks were
   explicitly deferred for now; document rather than invent a contrary scoring rule.
7. Each hour aggregate finalized epoch points and normalize each miner's count by
   the total. Define window boundaries/restarts explicitly. Zero total points means
   no new submission, not fabricated rewards or automatic owner burn. Respect
   chain normalization/cardinality/rate/version/commit-reveal constraints; report
   policy denials instead of padding fake qualifying miners.
8. Trainer on separate execution role consumes accepted batches, applies real
   training steps, saves immutable weights and verifies complete upload. Only then
   open the next epoch. Public audit history persists independently of training.

## Proof claim and conservative initial policies

User accepts curated/external tokens if approved-model computations on those
contexts verify; original seed/temperature sampling provenance is not required.
TOPLOC fingerprints intermediate activations; full reported conditional
probabilities require their separate forward-pass check. Validator reference model
hashes are external authority, never an untrusted artifact's self-declared policy.
No arbitrary code/pickle/remote-code execution from miner uploads. Bound ZIP sizes,
entries, tensor shapes, tokens, turn counts and proof counts; reject missing/null/
truncated/extra proofs and nonfinite probabilities.

Audit/slashing details and training objective were left open. For initial runnable
version choose a documented conservative full-audit policy and reject invalid
batches with zero contribution; no stake confiscation. Expose randomized auditing
as configurable only with an honest statistical assurance statement. Train on
fully verified data initially. Sequence preference loss is a provisional explicit
mock objective, not per-token credit attribution or finalized research policy.
K=L=1 and tiny Mastermind are smoke-test settings; expose larger K/L and environments
as config. Production model/environment selection must be versioned and explicit.

## Existing implementation and resources

- Working directory /home/const/subnet120-rewrite, branch
  rewrite/inference-verification; read AGENTS.md and STATE.md. Legacy source removals
  are staged in this isolated worktree, not applied to production. Preserve user
  changes and other agent files. No push is implicitly required.
- Partial new files subnet/storage.py and subnet/model.py are untested foundations
  written by root: actual R2 backend, local deadline-enforced signed upload gateway,
  Ed25519 sealed capabilities, CPU inference/proofs/replay and real preference
  training. You own completing/testing/refactoring these now. Root stops editing
  these files after handoff. Add controller/miner/verifier/scoring/trainer/CLI and
  meaningful integration/adversarial tests plus operational docs.
- Dedicated bucket ALREADY CREATED: affine-verification-mock-20260930.
  Nonsecret config state/mock-r2.json references existing credential file. Do not
  recreate production buckets or make the whole bucket public (would reveal active
  private uploads). Model publication/frozen data/public audit access may use public
  gateway paths or explicit signed downloads, clearly documenting their visibility.
- Credential locations (never echo values): /home/const/subnet120/.env has
  CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, R2_ACCOUNT_ID, R2_ENDPOINT,
  R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY, GitHub and Discord credentials.
  ~/.lium/config.ini has Lium API key. User-authorized Arbos vault contains Cloudflare
  credentials too; use existing authorized operator credentials when available.
- Validator wallet /home/const/.bittensor/wallets/default/hotkeys/default;
  name default/hotkey default; expected public hotkey
  5HmYnmUYT6qe3yFMg1Ad8WLLqDvnwtjYakXBpDvoRW1Qqzb8; Finney netuid120.
  Never output private key or commit wallet/env. User authorizes actual weight
  submissions for the finished system, subject to validated eligibility/policy.
- Local135M snapshot is described in prototype/model-source.json (ignored).
  Complete1.7B trusted model+artifact in prototype/artifacts/complete-rollout.tar.
  prototype/pipeline.py is the real TOPLOC baseline with honest/curated accepted
  and14 tamper cases rejected; reports and3.44GB artifacts locally saved.
  Its CUDA/model/environment settings are experiment-specific; do not blindly reuse
  fixed hard-coded indices/paths in a production epoch worker.
- .venv borrows archived package libs through .pth; torch2.14.0+cu130, transformers
  5.14.1, verifiers0.3.1. Root installed boto3/PyNaCl in this isolated env. Agent
  /root/verification_experiment is rebuilding TOPLOC0.1.6 against this ABI using
  private extracted Python headers, and running CPU-vs-GPU/spot-check tests. Coordinate
  through messages, avoid duplicate builds/edits in prototype area.
- Previous experimental GPU pod8c416640-e9dd-453a-b78f-ce1d9a866df8 disappeared;
  confirmed ledger user_initiated removal17:20:20UTC, charge$0.382979. Do not assume
  it exists. Mock can run CPU with capped4threads. If live operation requires GPU,
  provision one cost-effective suitable Lium pod within user-authorized infrastructure
  scope, record price/id/owner and retain it running (user requested retained pods).
  Do not interfere with old eval/bench/teacher resources.

## Existing production and transition safety

Original live code /home/const/subnet120; full private restorable archive under
/home/const/subnet120-archive/20260930T144912Z. Old validator/eval/bench still running;
chat provisioning disabled and chat removed per user. Do not terminate old resources
or erase legacy models/state. User announced previous competition transition max2days.

Owner-burn timer disabled. New equal-registration transition script
ops/equal_registration_weights.py and user timer affine-transition-weights.timer
watch new verified affine2 registrations after block9181759
(2026-09-30T15:54:34UTC), expiring2026-10-02T15:54:34UTC.
Marker ~/.local/state/affine-transition/active makes original validator
_maybe_set_weights return early while intake/eval continues. Deploy new payout
writer as SINGLE writer: stop transition timer before live new mechanism starts,
keep old-validator guard unless intentionally restoring old payouts. Inventory
active service state first and preserve a rollback procedure; do not create races.
PM2 intentional stops/restarts use ~/.affine/deadman.pause; do not use broad pkill.

## Validation milestones and definition of done

A. Persist plan/config and reproducible install/runtime (do not depend indefinitely
   on archived .pth). Unit checks meaningful scoring/epoch/crypto access invariants.
B. Actual R2 mock: three local Ed25519 miners, legitimate generated positive and
   negative model trajectories, encrypted capability isolation, cumulative updates,
   late-write rejection, private-before/public-after, frozen hash stability. Include
   overlapping index to prove duplicate-zero scoring and independent unique scores.
C. Separate verifier process loads trusted checkpoint and reports full inference+
   environment checks; invalid submissions cannot poison training or cancel valid
   competitor. Genuine training changes weights, publishes new checkpoint and
   creates next epoch referencing that exact hash. Verify next-checkpoint model load
   and at least one new miner rollout; record evidence, costs, paths and commands.
D. Live integration reads fresh chain registration/key ownership and maps final
   points onto CURRENT UID/hotkey pairs; protect against UID recycling and owner
   mismatch. Dry-run chain plan, observe rate policy then submit eligible nonempty
   weights using existing SDK commit/reveal; never fabricate miners to get a success.
   If no live miners participate, keep ready service waiting with no extrinsic and
   report this exact limitation (live accepted weights requires actual eligible
   contributions). Do not silently register paid test miners or use production
   miners' identities as mock identities. Mock demonstrably completes separately.
E. Service lifecycle, checkpoints/restart recovery, private state/per-miner keys,
   bounded upload/compute budgets, logs/health/audit history, explicit deployments,
   rollover/rollback and operator README. Final handoff accurately differentiates
   mock success, live readiness, live participation and accepted chain writes.

Send root regular concrete milestone updates. Ask only for genuinely missing
material choices/constraints; choose reversible conservative defaults and document
them. No Discord announcements unless separately requested. No extra inherited
permission barrier: effective session is full filesystem/network access.
