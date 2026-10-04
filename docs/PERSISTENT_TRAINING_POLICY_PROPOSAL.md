# Prospective persistent FP32 training policy

`bf16-cpu-fp32-master-task-normalized-persistent-v4` is implemented in dedicated
modules for review. **It is not selected by the live controller, staged immutable
source or any existing job.** No machine, chain setting, mining sampler, historic
reward or public mining contract was changed by this work. Production activation
requires a new source archive, signed contract and actual GPU qualification.

## Why this policy exists

The current BF16 AdamW policy can round nonzero small updates away. Keeping only
an epoch-local FP32 copy can still lose those updates when each export rounds to
BF16 and the following epoch reloads it. This candidate therefore preserves FP32
master values, FP32 first/second Adam moments and exact integer counters across
epochs, while every model forward and published inference checkpoint uses the
exact BF16 projection of the master. The preference reference remains the signed
BF16 input checkpoint, computed before any updates in that epoch.

The optimizer uses the current learning rate `1e-5`, betas `(0.9, 0.999)`, epsilon
`1e-8`, explicit weight decay `0.01`, clipping norm `1`, and mean-token preference
beta `0.1`. State transport binds these values; changing them requires a distinct
contract/migration. A new optimizer is **not** silently created when state is
missing, corrupt or unavailable.

Task weighting is also explicit: average losses of distinct verified pairs
within one `(environment, task index)`, then average those task losses within
each optimizer group. Exact copies collapse first. Several identities cannot
multiply one task's total gradient share by supplying more pairs for that task.
This does not change historical uniqueness rewards, and it does not prevent
selection of different valid on-policy pairs within a task. Root must decide
whether this rule becomes the future contract.

## New interfaces

`subnet.persistent_cpu_adamw.PersistentCPUAdamW` accepts named BF16 parameters,
the exact input inference checkpoint ID, and exactly one of:

- An explicitly authority-approved genesis document/digest binding policy,
  hyperparameters, parameter inventory and input checkpoint. Genesis/reset must
  be explicitly signed in the new job; missing previous state is not genesis.
- A restored authenticated parent-state tuple. Its inference checkpoint must
  equal the current model checkpoint, all parameter names/shapes/counters must
  agree, and every FP32 master's BF16 projection must equal the actual input
  parameter tensor before any training update.

The optimizer keeps CPU FP32 state and processes one parameter's gradient and
temporary denominator at a time. The normal training entry point clips the
full gradient according to the fixed rule before `step()`. The optimizer reports
actual FP32 changed elements/delta norms and actual BF16 changed elements for
every tensor. Nonfinite gradients/state, absent gradients, partial counters and
projection mismatches fail explicitly. BF16 gradient accumulation remains BF16;
this candidate does not claim fully FP32 forward/backward computation.

`subnet.task_normalized_training.train_epoch` accepts independently verified
pairs, an authenticated restored state or explicit genesis, the signed epoch
and coverage seed, and a real resource admission. It returns a BF16 checkpoint
directory, the updated CPU optimizer and diagnostics. It computes every input
reference before updating, covers every distinct task, records task-normalized
pair weights and checks pre/post training preference margins. The returned
`complete=False` is intentional: no next epoch is admissible until model bytes
and persistent state have been independently published and read back.

The caller must hash the actual BF16 export and use that ID when publishing the
state descriptor. **Updated masters with unchanged BF16 inference tensors are
legitimate.** Advance authenticated state/counters in that case; do not discard
the residual or pretend there was a held-out gain. Existing policies that reject
every unchanged inference checkpoint cannot be reused unchanged for this policy.

## Streaming state transport

`subnet.persistent_training_state.restore_state` takes an authority-approved
descriptor payload SHA, the current inference checkpoint, the actual parameter
inventory, an actual resource admission and `fetch_shard(name, destination)`.
The callback downloads exactly one object into a fresh private transfer folder.
The implementation hashes every byte, checks the exact safetensors allowlist,
FP32 shapes and complete nonoverlapping coverage of all three state slots,
materializes CPU buffers, closes the mapping, then removes that local shard.
File order is immaterial. Failed transfers remain for inspection and never
return partially restored usable state. No full ~91 GB state copy is required
on disk.

The fetch callback must enforce the signed object's byte limit while streaming,
including rejection of redirects or a changed object size before download.
Module-level digest/size checks reject corruption after fetch; a callback that
downloads an unbounded object first would defeat the disk admission contract.

`export_state` receives:

- `publish_shard(name, path)` for an immutable object in the current signed
  job's unique private storage namespace.
- `readback_shard(name)` yielding actual bounded byte chunks from that durable
  object. The implementation checks the full digest and size before deleting
  its local shard.
- `commit_descriptor(document)` that publishes and reads back descriptor bytes.
  It returns the payload SHA and `durable_readback_verified=True`. On GPU nodes
  this stages an unsigned descriptor and explicitly returns
  `authority_committed=False`. The coordinator separately hashes every durable
  shard and the staged descriptor, then signs and reads back the authoritative
  descriptor last. Only that authoritative commit can admit a next epoch.

The descriptor callback runs only after every shard's full durable readback.
Partial shard publication cannot admit a state or the next epoch. Use a unique
immutable storage prefix per signed training request; generated shard names
are local to that namespace and must never overwrite a preceding epoch's state.
The descriptor contains input/output inference checkpoint IDs, original genesis
digest, approved parent descriptor digest, all names/shapes, hyperparameters,
global/per-parameter counters and each shard's digest, size and tensor spans.
The existing signed job must bind its approved descriptor digest and namespace;
this module does not replace authority authentication with an unsigned JSON.

State is safetensors, never pickle or `torch.load`. Objects are capped at
4,000,000,000 bytes. Large parameter buffers split across shards; model exports
use a 3.9 GB target and enforce the actual 4 GB object cap afterward. One failed
object or descriptor acknowledgement preserves failure evidence and prevents
commit. Local files are disposable only after their matching durable readback
or successful authenticated in-memory restore.
Normal optimizer updates are locked out while state publication runs, preventing
a mixed snapshot if a storage callback attempts a reentrant update.

## Actual resource admission

Before allocating masters, observe Linux `MemAvailable` and applicable cgroup
memory headroom, plus actual workspace disk free space. Cgroup admission permits
only automatic reclaim of clean inactive file cache. It conservatively subtracts
all dirty/writeback, mapped, unevictable and shmem pages, counts no active file
cache, and remains capped by `MemAvailable`. Missing cache accounting permits
only hard headroom. No `drop_caches` or manual eviction is requested.
The required additional CPU RAM is:

```text
12 * parameter_count
+ 24 * largest_parameter_elements
+ maximum_transfer_object_bytes
+ RAM safety margin
```

The three FP32 state buffers alone require **91,387,398,144 bytes** for the
7,615,616,512-parameter model. Extra reserve covers bounded serialization,
gradient/denominator/delta/projection temporaries and an 8 GiB safety margin.
RAM is checked again before genesis allocation or streamed restoration. No claim
is made that the retained H200 boxes currently have this CPU RAM available.

The operator's recorded trainer observation was 248,710,017,024 bytes
`MemAvailable`, cgroup limit 248,034,361,344, charged usage 219,109,007,360, and
107,243,298,816 bytes free disk. The Qwen config has `hidden_size=3584` and
`vocab_size=152064`, implying a largest embedding/head of 544,997,376 elements.
For the measured 7,615,616,512-parameter model, this formula requires
117,057,269,760 bytes additional worker RAM. The coordinator adds the measured
BF16 input load and one raw-artifact reserve before loading the model; using
15,231,233,024 input bytes and a 3,000,000,000-byte raw reserve yields
135,288,502,784 bytes. The actual approved named-parameter inventory and
checkpoint file sizes, rather than these example calculations, control admission.
Clean inactive cache after exclusions determines whether this host qualifies.

Additional disk admission reserves one complete BF16 export, one transfer object
and an 8 GiB safety margin. It applies beyond the model and any input artifacts
already present. Do not include a nonexistent 90 GB disk state hydration, and
do not ignore accepted rollout files. V4 downloads one signed frozen ZIP at a
time, checks its full hash, independently reaudits exactly the admitted batches,
retains every training pair and report, then removes only that successful local
ZIP. Failure preserves the input. Coordinator admission includes a full raw
artifact reserve and one compressed ZIP, not the sum of all ZIPs. This avoids
retaining a potential ~88 GB input backlog on the ~107 GB disk. Actual available
RAM/disk still must pass on the trainer; no fit is assumed.

GPU forward/backward capacity remains an independent actual qualification gate.
CPU offload removes GPU FP32 optimizer buffers; it does not prove an H200 can
run every allowed prompt/trajectory or that the new implementation has numerical
equivalence on GPU. Preserve signed runtime/source/device/context limits and
run an actual honest-pair training/reload/state-round-trip control before live
promotion. Changing this contract must not relabel an existing GPU control as a
qualification of these new modules.

## Implemented integration; activation requirements

1. Pin all new policy/transport/protocol modules and any dependency/runtime profile in the new
   immutable source archive. Current covered v3 jobs remain unchanged.
2. Authority-sign a one-time explicit genesis, or bind the exact previous state
   descriptor SHA/namespace in every new training request. Preserve the input
   BF16 checkpoint, original frozen receipts and coverage challenge.
3. Admit actual RAM/disk and GPU capacity. Authenticate and stream parent state
   before training; never repair missing/corrupt state by zeroing moments.
4. Reaudit every selected training pair with the current forced-sampling,
   full-logit/TOPLOC and native environment checks. Normalize contribution by
   verified task, independently of reward uniqueness policy.
5. Publish/read back actual inference objects and state shards, then sign/read
   back the descriptor last. Bind output model/state to the exact original
   request and actual successful child completion before advancing controller
   training counters or opening the next epoch.
6. Recovery reuses the original completed model/state outputs. It must not
   double-apply an update or substitute a later state. Preserve failed jobs and
   partial publications; old approved state remains usable for a new explicitly
   authorized recovery attempt until a completed successor is committed.
7. Publish the exact future policy, sampler, training-state semantics, quotas
   and honest unchanged-BF16 behavior in GitHub and public `/llms.txt` together.
   Preserve the original independent held-out benchmark and its selection rule.

The prospective integration is present in `backend_jobs`, `remote_backend`,
`controller`, `gpu_service` and `role_router`. Its standalone helpers are
`persistent_training_protocol`, `persistent_training_worker` and
`persistent_training_controller`. Existing policies/defaults keep their old
paths; no current immutable source or live configuration selects v4.

Before opening v4, provide `persistent_training_admission` with the actual
approved `parameters` inventory, its `parameters_sha256`, qualified
`source_sha256`, `gpu_qualification_sha256`, one explicit `genesis_round`,
`genesis_checkpoint` and exact `genesis_sha256`. Missing prior state never falls
back to genesis. A successful state is journaled monotonically in
`latest-trainer-state.json` and bound into the next signed opening, even when
BF16 inference bytes have not changed. Signed training jobs bind exact parent
publication/namespace, genesis and before/after counters, plus job-scoped PUT/GET
capabilities. Root qualification must actually approve the inventory, source,
GPU profile, full admitted trajectory limits and resource measurements before
supplying these admission fields; a digest alone is not qualification evidence.

Worker reports and coordinator metrics retain `updates` as the actual ordered
optimizer-update list, with exactly one entry per signed training step. The
non-list state/precision summary lives separately in `persistent_diagnostics`;
cached recovery binds both to the original collected worker report. It also
requires the authority publication's exact original job ID, job SHA and output
namespace, rather than selecting another publication with an identical state
descriptor or BF16 checkpoint.

`subnet.persistent_training_evidence.validate_updates` is the dedicated pure
v4 bookkeeping checker called before authority state admission. It reconstructs
the task-normalized schedule from independently audited pairs, checks every
pair's contribution/reference and full task coverage, then binds per-update
global counters and FP32/BF16 precision diagnostics. It does not run inference,
claim historical GPU execution or claim held-out improvement. The legacy
`ops/check_gpu_continuous_evidence.py` also assumes cyclic single-pair attribution
and always-changing BF16 weights, so it must not be used to certify v4 merely
because the update-list shape matches. A whole-epoch v4 evidence route still
needs to combine this checker with signed model/state publications and the
original independent held-out evidence.

## Local controls and limits

```bash
.venv/bin/python -B -m unittest discover -s tests -p test_persistent_training_policy.py
.venv/bin/python -B -m unittest discover -s tests -p test_persistent_training_integration.py
.venv/bin/python -B -m unittest discover -s tests -p test_persistent_training_security_review.py
.venv/bin/python -B -m unittest discover -s tests -p test_persistent_recovery_review.py
```

Controls cover exact FP32 AdamW arithmetic, three nonzero master updates with
unchanged BF16 exports, persistent updates across four export/restore epochs,
shuffled streaming shard order, model/parameter/policy/counter/hash binding,
corrupt transfers, failed readback without descriptor commit, explicit resource
refusal, task-level gradient normalization and nonfinite refusal. They use actual
CPU torch and safetensors, with a local storage callback implementation.

These controls do not establish GPU capacity, remote R2 correctness, model
convergence or exploit freedom. The new policy must receive independent root
review, actual remote qualification and a complete signed source/protocol handoff
before it can replace the deployed mechanism.
