# Synchronous epoch system

`subnet.service` is the epoch controller. It trusts its own checkpoint file hashes
and chain-owned registration keys, publishes a signed manifest, and creates one
Ed25519 sealed upload capability per miner. Current direct-R2 transport gives each
miner a single-object presigned PUT and signed hash-pinned GET routes. The bucket
stays private. At freeze, one atomic GetObject obtains both bytes and R2 completion
metadata; only completion before the deadline qualifies. A private immutable
hash-addressed snapshot is persisted before publishing the frozen artifact. Late
in-flight replacements cannot change that snapshot or its score. Presigned URL
expiry alone does not enforce finality. Signed renewable audit-history routes make
frozen artifacts and reports readable without operator credentials or a tunnel.
The older gateway transport remains supported separately and serializes uploads
and freeze under a lock. Keep private operator state mode0700.

Miners download and check an exact checkpoint file allowlist, then use a signed
registry of trusted `EnvironmentSpec` entries and separately versioned harnesses.
Each specification pins source/configuration, dependency versions, task data when
snapshotted, environment indices, reward classification and resource budgets.
Adapters expose reset, public observations/tools, action execution and terminal
reward. The core miner/model/verifier/trainer does not select environment names.
Uploaded artifacts never select executable code or a reward function.

Harness implementations own rendering, action parsing, observation handling and
sampling. `text-tools-v1` uses the tokenizer's chat template and a declared JSON
text tool protocol; `plain-transcript-v1` provides an explicit role transcript.
Tools execute through trusted original runtimes, including actual sandbox tool
calls where supported. Native tool observations remain in the audited trace and
are rendered as text observations for this portable harness. Unsupported wire
protocols fail explicitly. This is not an assertion that every upstream harness
or all45 active sources have completed an end-to-end run: consult the execution
matrix and original dependency/resource evidence in LEGACY_ENVIRONMENTS.md.

Autoregressive policies genuinely sample target-model tokens; candidate policies
are curated controls chosen by target-model likelihood. Signed bounded per-turn
policy overrides allow a tool action followed by a final response without adding
special cases to core model code. None of these policy implementations reads
hidden task answers. Curated controls establish plumbing, not autonomous solve
performance. Every turn carries exact context/output tokens, full conditional
log-probabilities, TOPLOC fingerprints, structured observations and graded outcome.
The current pilot requires K=L=1 and bounded small task sets. A separate signed
held-out contract can use an autoregressive harness with fixed indices/seeds;
held-out task sets remain disjoint from training submissions across checkpoints.

At closure the controller freezes each final private file and publishes its SHA256
receipt. Separate verifier subprocesses recompute all model distributions and exact
TOPLOC fingerprints and replay every turn using the approved original environment adapter and task data.
Signed audit reports refer to frozen hashes. Invalid batches receive no points and
cannot cancel another miner's valid batch. Valid `(checkpoint, environment_id, sample_index)` keys present in
multiple miners' batches earn zero for everyone; one valid unique key earns one
point. Accepted duplicate batches may still be training data; scoring uniqueness
and valid training eligibility are separate.

The initial verifier uses **full audit**, CPU float32/eager inference and exact
TOPLOC matching plus1e-5 absolute log-probability tolerance. CPU/GPU/precision/runtime
portability is NOT established: originalGPU proofs fail honest CPU checks and broad
relaxation admitted attacks. Miners must use the advertised runtime. Random audits,
statistical rewards and punishment schedules remain research choices and are not
silently enabled. Configurable seeded post-freeze audits exist for research;
unchecked samples never train, and subset scores explicitly remain provisional
with incomplete duplicate coverage. Penalty means rejecting contributions, not
stake confiscation.

`subnet.trainer` is a separate subprocess/execution role. It receives only the
controller's verified positive/negative pairs and performs a genuine reference-
relative sequence-preference gradient update. This is a provisional smoke objective,
not finalized per-token credit assignment. The checkpoint upload is checked file
by file before publishing the next epoch. No verified pairs pauses the synchronous
loop; it does not fabricate training or owner rewards.

The chain adapter maps signed on-chain Ed25519 activation keys to fresh UID/hotkey
pairs, rechecks ownership at submission, and submits normalized hourly totals only
if SDK policy allows. An empty score window produces no extrinsic. Legacy payout
and transition writer exclusion must be established before enabling this writer.
History is immutable public epoch manifests/checkpoints/submissions/audits/scores/
training reports; private local state contains capability secret, authority key,
checkpoint paths and restart phases. Do not commit private state.

## Commands

Run the complete genuine mock (no blockchain):

```
.venv/bin/python -m subnet.mock --bucket-config state/mock-r2.json
.venv/bin/python -m subnet.tests
```

Start a registered miner:

```
.venv/bin/python -m subnet.cli --gateway https://YOUR_GATEWAY --authority AUTHORITY_HEX --key /private/miner.seed
```

Separate verifier/trainer/controller CLIs are exposed by `pyproject.toml`.
See LIVE_SUBNET.md for chain single-writer cutover. Source checkout is required by
this initial version because trusted pinned upstream source snapshots live in
subnet/vendor and prototype/vendor. Exact credential-free historical verifier
source bundles are retained under state/source-bundles and publicly hash-addressed.

## Dependencies and limitations

Install Python3.12 and project dependencies from pyproject.toml. TOPLOC0.1.6 must be
compiled against the exact torch ABI; the supplied experiment initially required
rebuilding its C++ extension. Runtime here borrows archived package libraries using
`.venv/.../archive-dependencies.pth`; it is isolated from writes to production but
is NOT a portable independent installation. Before moving hosts, recreate packages
and rebuild TOPLOC, then rerun honest/tamper tests. Current torch2.14.0+cu130,
transformers5.14.1, verifiers0.3.1, TOPLOC0.1.6.

Gateway serializes writes globally, re-uploading cumulative ZIPs is bandwidth-heavy,
and full-vocabulary probabilities are large. Upload limit100MB, decoded tensor/archive
budget500MB, bounded turns/tokens/proofs and process timeouts are initial operational
budgets. Scaling requires sharding/storage and a validated sampling audit policy.
