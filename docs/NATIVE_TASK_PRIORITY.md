# Prospective native task priority and optimizer threading

The controller update was installed on October 12, 2026 at 04:27 UTC, scheduled
from epoch 158. Epoch 157 retains its original contract; first-epoch results are
still pending. The signed manifest and execution release determine activation.
Existing signed epochs keep their original policies. Neither
change requires a miner client upgrade or changes the public sampling contract.

## Prefer tasks missing from recent training

Task cooldown and native admission address different parts of coverage. The
existing cooldown changes which task indices are available for a future epoch.
The prospective native selector orders the committed candidate tasks before
running the original native grader:

1. Read the preceding 12 finalized epochs and attribute tasks to the optimizer
   updates that actually consumed them.
2. Freeze the authenticated history and candidate population before seeing the
   current native grading results.
3. Put candidate tasks absent from those 12 epochs first, preserving the existing
   deterministic task rank within each group. Recently trained tasks follow.
4. Keep the existing candidate order within each task. If a candidate fails or is
   unresolved, try the next candidate for that task under the original limits.
5. Train on the accepted native subset, including a nonempty subset below the
   task cap. This policy does not make full proof audits a training barrier.

The manifest opt-in is `training_representative_policy.version =
"first-native-valid-distinct-task-v3-prior12"`, with `history_epochs: 12` and
`task_priority: "untrained-in-prior12-first-stable-rank"`. See
`subnet/training_task_representatives.py` for the exact schema and version names.
The policy does not invent submissions for missing tasks or guarantee broad
coverage when miners supply a narrow set. It changes admission priority, not
reward rules, native grading, quota requirements, model math, or sampling draws.

History includes signed closure/job/input provenance and linked trainer reports.
Trainer reports remain operational evidence, not cryptographic proof of training.
Unavailable or ambiguous history freezes an explicit original-order fallback.
An already frozen history is reused on retries; corruption is an integrity error,
not permission to silently choose another history. Reads are bounded. The archived
draw and derived native receipt bind the exact history hash for replay and audit.

The public miner still follows the signed `environments[].indices`, checkpoint,
deadline and sampling contract. K4/L4, eight distinct completed rollouts per batch,
nine batches per UID per epoch and nonces 0–999 are unchanged by this patch.

## More CPU threads only during the optimizer step

The prospective trainer execution release adds an explicit signed policy:

```json
{"version":"optimizer-only-cpu-threads-v1","steady_intraop":2,"optimizer_intraop":16,"interop":2}
```

A single exclusive trainer worker keeps two intra-op threads for the existing
forward/backward and gradient clipping path. Immediately around the existing CPU
optimizer `step`, it switches to 16 intra-op threads, then restores two in a
`finally` block. Inter-op threads remain two. This changes thread scheduling;
it does not change the objective, learning rate, task weights, gradient clipping,
Adam implementation, optimizer counters, or checkpoint format.

Preparation, the signed training job, worker admission and independent report
validation bind the policy and exact source. Every update must report the
before/during/after thread counts. Historical releases without the opt-in retain
the original behavior and cannot claim the new evidence. The independent
basic-eligibility peer has a distinct v11 authorization for this v6 execution
release; its v10 path retains its original 186-file/v5 contract.

On one retained full 7B optimizer state and gradient, the isolated comparison took
172.1 seconds with two threads and 54.2 seconds with 16. All 339 parameter/master/
moment/projected-state records agreed. That is evidence for this mechanical
optimization, not a universal cross-hardware equivalence or learning result.
Actual worker CPU allocation and the source-bound caller must still be qualified.
If the optimizer throws, threads are restored but partially updated tensors are
not rolled back; that state must not be published or blindly retried.

## Checks and rollout criteria

Portable CPU tests exercise historical and prospective policy parsing, signed
history replay, checkpoint/counter discontinuities, native-selector retries,
fallbacks, input admission, signed execution preparation and report validation,
and thread restoration on success and failure. They do not run a model, grade real
answers, or issue production signatures.

Before activation, qualify the actual coordinator/preparation/CPU-peer/backend
route against the admitted source. Preserve in-flight epochs and optimizer state.
After the controller handover, check the public `/mining.json` against the signed
current manifest as well as checking service health. Its publisher must recognize
the composed policy and authenticate both the physical entry and imported runner.
A running discovery service alone does not establish working miner discovery.
After activation, compare actual trained-task novelty, native rejection rates,
external submission yield, optimizer and full-epoch duration, and held-out
performance. Broader task coverage and faster optimization are hypotheses to
measure, not proof that the model has improved.

## Worker startup correction (October 12)

The first epoch158 attempt stopped before optimizer construction: the worker set
intra-op threads to two but left inter-op threads at the runtime default (96 on
that host). The signed CPU16 release correctly refused this 2/96 entry profile.
The parent optimizer remained at step67; there was no completed numerical update.

The peer now initializes the admitted steady 2/2 profile after authenticating the
job and peer grant, before importing the backend or doing inter-op work. Its
initialization receipt records both thread counts. Historical jobs without the
explicit threading declaration retain their behavior. A reused interpreter that
cannot set inter-op threads still fails; the worker must start in a fresh process.
The optimizer's existing 2→16→2 checks and per-update report requirements remain.

Restart also exposed an older controller startup guard that only recognized the
prior v2 task-selection policy. The operational successor keeps that guard's code
and historical behavior, while binding its prospective comparison to the exact
ROOT-signed v3 policy from epoch158. It does not rewrite the issued manifest. The
complete composed startup path was checked to the first operational-client
boundary, with network, signing and mutation disabled, before activation.

A separately authorized replacement job began at07:04 UTC with the same signed
manifest, all512 selected input descriptors, checkpoint and optimizer parent.
Its initialization receipt confirms 96/96→2/2. Original failure evidence and the
epoch start time remain intact; epoch158 has missed the one-hour target. This
record does not claim a completed replacement update or improved model quality.
No miner upgrade, sampling-contract change, or website redesign is required.
