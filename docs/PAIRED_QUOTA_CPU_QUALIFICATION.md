# Prospective paired-quota CPU controls

`ops/paired_quota_qualification.py` is a CPU research helper, with **no production
call sites or activation**. It supports the controlled experiment in
[PAIRED_SAMPLE_STABILITY_PLAN.md](PAIRED_SAMPLE_STABILITY_PLAN.md). Default quota
is one success and one failure. K2/L2 is only an explicit helper argument.

An authenticated adapter must supply the pinned task, checkpoint, taskset,
harness, sampling context and approved prescribed attempt IDs. The helper
recomputes execution and content identities; submitted IDs are never trusted.
Execution IDs distinguish approved attempts, while content IDs exclude miner,
attempt, classification, reward, filenames, proof wrappers and upload time.
Content includes all ordered prompt/output tokens and actions/observations.
The epoch belongs to the contribution slot; the approved sampling-context digest
belongs to execution. Checkpoint/taskset/task/harness bind both identities.
Canonical identities are namespaced SHA256 hashes of strict JSON metadata.

The whole cumulative task-slot revision is checked before selecting pairs.
Exact redelivery adds no member. A repeated execution with changed content or
label is refused; repeated content from a different attempt is a duplicate and
cannot fill more quota, **not a fraud verdict**. Conflicting classifications
cannot turn one content into both success and failure. Pairing is deterministic,
independent of upload order, and uses each selected member once. K2 produces two
nonoverlapping pairs, each with half the task's weight. Reward contribution
remains one task unit. Cross-UID copies preserve content identity while the
miner-specific task slots differ; the helper does not itself assign cross-miner
zero scores.

These are cheap structural controls, not proofs of authentic sampling or native
outcomes. The actual prescribed draw and inference verifier still has to check
every audited trajectory; grading still authenticates outcomes. Supplying four
synthetic traces with plausible labels to this helper alone is not a qualified
sample-generation test.

## Remaining integration and research evidence

- The production rollout schema uses harness-specific fields. Write and qualify
  an adapter that extracts complete canonical actions/observations and verifies
  approved context/attempt bindings from actual manifests. Do not just trust
  unverified caller metadata or drop tool turns.
- Recompute identities across the entire cumulative commitment, bind selected
  revision/content IDs into signed training inputs, and freeze that revision.
  Canonicalization must remain consistent across miner, eligibility and audit.
- Persist a selected task-slot and training-job association with transactional
  uniqueness, preserve completed receipts on recovery, and prove actual trainer
  restart/redelivery cannot apply the selected revision twice. The returned
  stable `revision_id` is **not** a persistent ledger and does not implement
  optimizer idempotence. These tests establish only repeat-selection identity.
- Preserve existing cross-miner same-task duplicate-zero scoring and audit
  evidence deduplication. Neither is implemented by this helper.
- Qualify the existing producer on real K2/L2 attempts, then run matched
  same-parent/Adam K1/L1 and K2/L2 branches. The CPU tests say nothing about
  held-out gains, GPU/runtime compatibility or achieved task coverage.
- Activate only through a reviewed future contract, published miner guidance,
  adjusted budgets and a real end-to-end epoch. No live configuration changes
  are made here.

CPU controls cover four distinct members, task normalization, repacking,
reordering, label conflicts, repeated attempts, repeated content, cross-UID
copies, old checkpoint/epoch refusal, exact redelivery, complete trace bindings
and schema/budget refusals. Run:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B -X pycache_prefix=/tmp/paired-quota-fresh-review \
  -m unittest discover -s tests -p test_paired_quota_qualification.py -v
```

Use a fresh cache prefix for each qualification run. This adds neither a live
proof exemption nor a deployment authorization.

## Current wire-format research adapter

`ops/paired_quota_batch_adapter.py` adapts the current `schema: 2` batches
produced in `subnet/backend_jobs.py` and rollouts produced in `subnet/model.py`
and `subnet/gpu_runtime.py`. It remains research-only with no production call
sites. It reads epoch/checkpoint from the enclosing batch; rollout `seed` is the
prescribed attempt, with its receipt independently recomputed through
`forced_sampling.binding/receipt`. It resolves the authorized per-index harness
through `protocol.entry/harness_for`, refuses held-out indices and old checkpoint
wrappers, and checks environment version, task hash and environment seed.

The caller must authenticate the manifest and provide an independently obtained
native-reset task hash and pinned tokenizer decoder. The adapter derives actions
from decoded output tokens through the actual harness action parser and checks
claimed text against that decoder. It keeps every turn's ordered observation
role/content, including tool observations; nonsemantic proof wrapper fields are
excluded. This does not verify that prompts or observations are truthful: the
independent native environment replay still must do that.

`CumulativeTaskSlot` accepts successive cumulative snapshots of one fixed
miner/task slot. Re-delivery, reordering and proof repacking retain the same
revision identity. A later snapshot cannot remove or rewrite an existing attempt;
all checks precede state mutation, so refused updates leave the previous snapshot
intact. Distinct attempts with repeated content remain quota duplicates. Identical
prompt/output traces with contradictory observations are refused as inconsistent
metadata, rather than becoming extra quota. This is not a fraud determination.
It retains one unit per qualifying task and one pair by default; explicit
`quota=2` selects four distinct members without reuse.

These additional controls exercise **synthetic current-format fixtures**, not
original miner executions. They do not load model weights, run TOPLOC/CDF,
replay native grading, activate K2/L2, or implement persistent job receipts.
The checker is deliberately in-memory: it does not survive process restart and
must not be treated as the production optimizer idempotence barrier. Actual
receipt/transaction integration and restart/recovery tests remain required.

## Opt-in durable research selection ledger

`ops/paired_quota_research_ledger.py` adds an explicitly enabled, privately
created SQLite research ledger. It remains unconnected to production callers.
A transaction binds the complete selected revision population to one original
job ID, settings digest, parent optimizer-state descriptor digest and starting
step. Task slots, revisions, execution IDs and content IDs have uniqueness
constraints. Reservation rollback is all-or-nothing, including conflicts late
in a multi-slot reservation. Repacking/redelivery cannot reserve or count the
same selected data in another job; copied members across UIDs also conflict.

Before invoking the executor, the ledger durably changes `reserved` to
`executing`. Only one concurrent claimant can invoke that executor. A completed
job returns its recorded original receipt on redelivery without invoking it
again. Exceptions change an unresolved claim to `uncertain`. A hard process
crash leaves `executing`; neither state has a timeout, reset or automatic retry.
This deliberately trades liveness for avoiding uncertain double application.

The crash tests actually terminate child processes before or after writing a
**simulated** optimizer-step marker, reopen the SQLite ledger, and prove no
second executor invocation. They do not run a GPU optimizer. After-update
recovery records the same original result only through a caller-supplied outcome
authenticator; tests use a synthetic authenticator. Real integration must verify
the original signed job/report, exact selected input population and parent,
optimizer step lineage, durable model/state objects and independent readback.
A receipt's self-declared success is insufficient. The ledger neither publishes
checkpoints nor mints authority signatures.

### What this does not guarantee

A SQLite selection claim cannot atomically commit a GPU optimizer update,
bucket publication and coordinator state pointer. A crash after the update but
before receipt persistence is indistinguishable here from a crash before the
update. Recovery must inspect the **original execution's** authenticated durable
outcome. If that evidence is missing, leave the job unresolved rather than
reapply it. Nor does this helper implement global compare-and-swap of the latest
optimizer parent, worker authorization, remote execution fencing, recovery from
loss of the ledger disk, or network receipt authentication.

Production already has separate admission/publication semantics:
`subnet/training_receipts.py` authenticates original completed verifier queue
records; `subnet/persistent_training_worker.py` stages state with authority commit
still required; `subnet/persistent_publication.py` gates publication on verified
original state/checkpoint paths; and `persistent_training_controller.commit_latest`
requires the expected parent and monotonic optimizer step. These are distinct
checks. This prospective ledger must bind to those actual original-job and
publication mechanisms instead of replacing them with a local `complete` flag.
Unaudited eligibility remains unaudited; this adds no inference exemption.

Still required before production: shared authoritative storage/backup and remote
fencing, selected-revision binding into signed job inputs, the actual original
publication/parent-CAS authenticator, and end-to-end trainer crash/recovery tests
with model/Adam state and durable readbacks. Research K2/L2 remains disabled.
