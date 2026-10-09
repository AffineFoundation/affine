# Explicit trainer state retirement and pre-update recovery

These CPU-only operator helpers manage local optimizer state. They do not change the model, sampler, optimizer mathematics, miner protocol, or resource reserves. Deploy them through an independently authenticated operator entrypoint; keep the three file hashes synchronized. The source map for scientific execution remains separate.

`trainer_reset_entry.py --operation plan` builds a read-only reset proposal from an already signed first training job and authenticated original ACK. ROOT authorizes the exact proposal before `--operation reset` can retire its old optimizer shards. The reset holds state and retention leases, fences both catalogues first, preserves original job/report/ACK evidence, and resumes interrupted deletion only for the same signed plan. `--operation check` is read-only.

`install_in_child(..., role="train")` runs inside the actual worker interpreter after source loading. `role="retirement"` runs in the actual ACK interpreter. Install again if the worker refreshes its module namespace. The post-ACK path promotes only an authenticated real candidate and then calls normal owned-checkpoint retirement; old-genesis ACKs cannot reactivate retired state.

A pre-update resource failure can be retried only with a separate ROOT grant binding both signed jobs, the unchanged manifest and selected inputs, source/runtime/effective learning rate, exact original terminal status and failure log, and the original reset. Training must still have no current or pending optimizer, candidate, or completed report. No second reset occurs. The retry entry accepts `recovery_path=` explicitly; for the existing retirement API only, it can discover the exact sibling `retry.ROOT-SIGNED.private.json`. An invalid sibling fails closed.

No private grants, keys, rollout artifacts, model weights, or machine-specific inventory belong in this directory. See the accompanying reset, recovery, and entry tests for interruption, expiry, changed-input, duplicate-job, promotion, and delayed-ACK cases.

`owned_input_page_advice.py` preserves the original checkpoint authentication
and resource guard. It advises only the exact owned model files after loading,
including a validated prior-job export cache, and records RAM before/after.
Install it in the actual worker process; a separate subprocess cannot change the
worker's imports. A source namespace reload requires reinstalling the hook.

`pre_update_retry.py` permits one explicitly ROOT-authorized replacement of an
original failed first-step job only when its authenticated traceback proves the
failure preceded optimizer construction. It preserves the original job, manifest,
inputs, source and reset record and binds a separate immutable retry grant.
