# Independent cached native evaluation

`ops.continuous_owned_cached_evaluator` is a CPU observer and dispatcher for a
separate nonpayable diagnostic queue. It uses the already qualified immutable
4db GPU source, its signed source route, and the explicit
`owned-cached-native-evaluation-v1` job policy. Updating the CPU orchestration
does not change that GPU source or activate the compact miner contract.

The original evaluator finished its checkpoint-nine full-proof job after
86.6 minutes. It remains a different original experiment. A cached checkpoint-ten
BEFORE job uses the same fixed 32 task indices and seeds, with an explicit
128-token cap. The observer queues AFTER only once the production controller's
signed completion, committed checkpoint, optimizer pointer and public optimizer
counter agree. The existing controller commits after its mandatory independent
full optimizer readback; this observer does not manufacture completion or ACK.
It then continues observing later genuine commits with the same diagnostic
policy. A missed checkpoint is not relabeled as a measured checkpoint.

ROOT signs the config and the one-source route, installs the authority in the
new local diagnostic state, and starts exactly one observer while keeping the
legacy evaluator scheduler stopped. The remote route uses a new workspace on
the independently idle evaluator. Original production jobs and queue files are
preserved. The observer verifies the old original process identity is terminal,
checks physical GPU occupancy before each new dispatch, holds its local process
lock, and retains exact issued identities after observation timeouts or failures.

The config includes `dispatch_allowed`, `source_sha256`, `source_bundle`,
`evaluation_mode`, `owned_evaluation_policy`, `evaluation_source_routes`,
`state`, `production_state`, `before_original_signed_job`, its full file SHA,
`before_checkpoint`, `before_optimizer_steps`, the original evaluator workspace
and job ID, `legacy_evaluator_scheduler_must_remain_stopped: true`, the fixed
heldout suite, model/environment config, bucket config and record paths.
Policy and source must be explicit; a ROOT-reviewed CPU package pins the
observer, `gpu_service.py` and `checkpoint_evaluator.py`. Run with
`python -m ops.continuous_owned_cached_evaluator --config ROOT-SIGNED.json`.

CPU evaluation routing now validates and passes the owned policy instead of
ignoring it. Unknown, null and mixed policies fail. Absent-policy queue
fingerprints remain unchanged. Policy-bearing fingerprints and dataset IDs are
distinct from legacy and unchanged-generation trusted-native jobs; original
requests require exact policy matching. Native grading is labeled `verified:
false`; infrastructure failures leave mean reward and uncertainty absent.

After an authenticated terminal report, the CPU observer publishes the exact
original signed job and report in a private R2 ACK and verifies its full-byte
readback. A CPU-only helper then retires that evaluated model from the new
owned workspace, before admitting the next checkpoint. It verifies the frozen
runtime hashes, original terminal/process absence, complete hydration receipt
and model map, and uses the existing single-link ownership/inode checks and
nonblocking inherited checkpoint lease. A busy lease or changed inode defers
cleanup. External mapped caches, original production namespaces, reports and
diagnostics are retained. This avoids retaining 15 GB after a completed
evaluation on a node with only about 26 GB free.

The genuine H200 qualification measured two 128-token tasks per arm: full-proof
41.56 seconds, unchanged-generation native 27.88 seconds, and cached native
10.33 seconds. Tokens and grades matched on those two tasks; both grades were
zero. These numbers establish compatibility and small-cohort performance,
without claiming a 32-task result or convergence. Learning comparisons must use
the same cached policy, source, model profile, task hashes, seeds and token cap
for BEFORE and AFTER.

The explicit `owned-cached-fixed32-1024-pair-v2` profile uses the same fixed32
indices and seeds with `evaluation_token_cap: 1024`, a distinct experiment ID
ending in `-cap1024-v1`, separate state/workspace and public epoch keys. ROOT
signs the complete config and source route. `before_optimizer_steps: 10` and
`after_optimizer_steps: 11` bind the original CP10 manifest to the immutable
ROOT-signed CP10-to-CP11 completion, exact AFTER checkpoint file map and durable
optimizer pointer. Later production checkpoints cannot replace that AFTER.
The route must bind an evaluate job TTL of 1800 seconds; observation timeout
continues observing the original signed job rather than issuing another job.
`stop_after_pair: true` exits only after both exact queue requests complete and
both genuine durable ACK model disposals complete, including lease release.
This is a separate cached1024 experiment; it does not convert either the
128-token latency diagnostic or the historical uncached1024 experiment.

`continuous-owned-cached-checkpoints-1024-v3` keeps the same scientific1024
experiment and fixed32 seeds, but has a distinct CPU config/service/state. It
requires `stop_after_pair: false`, the explicit1024 cap and1800-second original
job budget. ROOT pins an exact allowlist of the two completed v2 requests,
original signed jobs, authenticated reports and completed durable ACK disposals.
The initializer verifies hashes, ROOT signatures, full R2 ACK bytes, exact
policy/source/cohort/seed bindings and the frozen backend report validator before
copying original bytes into the new state. It never imports an authority seed,
changes the old state, relabels a report or resamples CP10/CP11. Scientific queue
fingerprints remain their originals; the CPU configuration has its own version
and hash. The canonical owned remote workspace is retained so original ACK
ownership remains exact. Start only after v2 original terminals, cleanup and
service exit are confirmed; every future dispatch still checks physical GPU idle.

Future AFTER requests follow only the controller's signed durable completion,
actual committed optimizer pointer and current checkpoint. A new optimizer12
request has its own checkpoint fingerprint. Evaluation remains asynchronous and
cannot become a trainer audit or heldout barrier. Automatic full-ACK model
retirement runs before subsequent disk admission. Unknown/null/mixed policy,
altered history bytes, wrong workspace, future-parent substitutions and partial
disposals fail closed. The genuine cached1024 CP10/CP11 pair measured19/32 then
18/32, one gain and two losses, with no infrastructure failures. Training margins
increased but this cohort did not establish heldout improvement or convergence.
