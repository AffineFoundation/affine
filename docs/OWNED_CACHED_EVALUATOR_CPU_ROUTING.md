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

The genuine H200 qualification measured two 128-token tasks per arm: full-proof
41.56 seconds, unchanged-generation native 27.88 seconds, and cached native
10.33 seconds. Tokens and grades matched on those two tasks; both grades were
zero. These numbers establish compatibility and small-cohort performance,
without claiming a 32-task result or convergence. Learning comparisons must use
the same cached policy, source, model profile, task hashes, seeds and token cap
for BEFORE and AFTER.
