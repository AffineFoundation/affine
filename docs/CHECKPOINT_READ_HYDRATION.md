# Read-only checkpoint preparation

`ops/checkpoint_read_hydration.py` prepares a complete, immutable model cache
from an operator-signed plan. The checkpoint ID is the hash of its full file
inventory. Every object's byte count and SHA256 are checked before the cache
becomes available. This supports new checkpoint IDs and shard layouts without
editing a helper for each training update.

The operator plan pins the intended node and role, checkpoint authority,
descriptor envelope, file sizes, exact HTTPS R2 read capabilities, destination,
helper hash and a window of at most one hour. It grants no bucket credentials,
publication writes, GPU work, optimizer execution or chain transactions.
Private plans and signed URLs must stay outside Git and public documentation.

Preparation may run concurrently with scientific work because it writes only
its isolated cache and receipt. It does not assert that GPUs are idle or that
the controller is held. Deployment still requires the separate completed-epoch
barrier and source, tokenizer, native-grader and runtime admission for the
actual successor checkpoint.

Run the approved helper with the node's existing Python environment:

```sh
python -I -B checkpoint_read_hydration.py \
  --plan /private/plan.signed.json --plan-sha256 PLAN_FILE_SHA256 \
  --operator OPERATOR_PUBLIC_KEY --role train --retained-UUID NODE_ID \
  --destination /private/checkpoints/CHECKPOINT_ID \
  --output /private/new-receipt.json
```

A kernel lock prevents concurrent hydration of the same local checkpoint.
An existing cache is reused only after checking every file, size and hash.
Downloads retain partial bytes, require exact HTTP range responses for resumes,
reject redirects, and never truncate a mismatching existing cache or partial.
Failure or expiry preserves the original evidence. Observe the original process
before deciding whether any new attempt is warranted; a polling timeout alone
does not justify starting another job.

Seven tests cover signed actor/scope/deadline binding, different checkpoint
layouts, descriptor tampering, URL/file scope, changed cache bytes, preservation
of incorrect partials and actual kernel-lock exclusion. These tests do not
replace full byte readback and actual child completion on each deployment node.
