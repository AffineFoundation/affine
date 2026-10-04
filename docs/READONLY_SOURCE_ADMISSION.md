# Prepare upgrades while mining continues

Stopping the controller to prepare every source, checkpoint and native grader
can consume the entire deployment hold before an upgrade is ready. CPU preparation
and live deployment now have separate evidence. Mining continues during preparation;
actual deployment still requires a completed epoch and fresh checks of every role.

`ops/readonly_source_admission.py` accepts an operator-signed, time-bounded request
for one role and a complete source/checkpoint inventory. It checks the complete
installed source before importing candidate modules, checks every checkpoint
byte, uses the local pinned tokenizer, and runs isolated original native grader
controls. It checks the source and checkpoint again before writing a new private
receipt. CUDA devices must be hidden from this child. No model is loaded, no GPU
job runs, and no live configuration or chain state is changed. The role interpreter's
packages are inspected rather than upgraded; native grader preparation uses the
environment's isolated pinned runtime.

The signed payload uses revision `immutable-readonly-source-admission-v1`. It binds
the role, helper hash, creation/expiration times (at most one hour), full source
descriptor and file hashes, checkpoint ID and file hashes, package versions,
dataset snapshot, native environment specification, harness, tokenizer/model
widths, context indices and native control indices. GPU jobs, chain transactions
and live configuration writes must explicitly be integer zero. Concurrent mining
must explicitly be permitted. Each invocation and receipt has its own namespace.

Run the helper in a fresh CPU process with private operator-created arguments:

```sh
CUDA_VISIBLE_DEVICES='' python -I -B /absolute/readonly_source_admission.py \
  --plan /private/plan.signed.json --plan-sha256 "$PLAN_SHA256" \
  --authority "$OPERATOR_AUTHORITY" --role "$ROLE" \
  --source /absolute/immutable-source --checkpoint /absolute/immutable-checkpoint \
  --archive /private/source.tar.gz --output /private/new-receipt.json
```

Preparation does **not** claim that other processes or GPUs are idle. It is not
an inference proof, hardware qualification, activation approval, or evidence of
learning. Original supervisors retain actual child-wait results and process
identities. An unavailable observation must be retried against those same
handles; it does not authorize another scientific run.

Deployment separately verifies source approval, the current completed checkpoint,
actual queue/job quiescence, runtime/hardware profiles, original process identities,
and the remaining expiring hold. A preparation receipt for an older checkpoint
does not establish that a successor's weights or metadata are correct. Preserve
the original evidence and admit the actual deployment inputs.

Six controls cover authorization, deadlines, role/helper binding, CPU scope,
checkpoint identity, altered or extra files, symlinks, and refusal to import an
unverified candidate. During the October 4 live epoch, the original preparations
on the miner, trainer, evaluator and first verifier completed while mining stayed
open: each checked 34 contexts and eight native positive/negative controls.
The second verifier's separate preparation also completed with 34 contexts and
eight controls, using the hardened pre-import helper. All five original CPU
supervisors exited successfully and their process identities were checked absent.
Original helper versions and receipts are retained separately. These are CPU
preparation results.

The separately installed recovery source also passed an actual H100 generated
rollout, model reload and environment replay. Token, proof, full-probability,
reward and checkpoint-hash mutations were rejected. This checks that specific
source/runtime/control; it does not qualify every hardware configuration or
prove an improvement in model performance.
