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

`training_ack_ordering.py` waits for the exact published state’s authenticated
promotion and completed retirement before calibration can use that same trainer.
It joins an existing ACK action and confirms the real promoted head; a locally
saved “complete” status alone is insufficient.

`terminal_ack_retry.py` can retry only the same signed ACK after an explicitly
deferred action has actually terminated and the workspace is idle. Retry handles
preserve the original result and payload. Unknown, failed or live actions never
trigger a new retirement process.

`authorized_lr_evidence.py` interprets the update’s hyperparameters from the
original ROOT-signed job and learning-rate grant, after validating the exact
output descriptor and diagnostics. It runs the unchanged attribution validator
with a private per-call expected-hyperparameter namespace. Legacy jobs retain the
original rate; global constants, reports and scientific execution stay unchanged.

`opening_assessment_ordering.py` delays acquisition of a new training blacklist snapshot until the existing successor calibration and optimizer ACK checks finish. It is installed only by an explicit future-round operator policy; issued openings retain their original snapshot and all signature, writer policy and freshness checks remain unchanged.

`fair_token_capture.py` installs a signed future-round capture policy:16 bounded I/O workers,128-document global checkpoints while retaining fsynced per-document recovery, postcommit random ordering, and parallel immutable parent copies. The implementation is in `subnet.training_documents`; its default legacy call paths remain unchanged until explicitly configured. The coordinator persists the seed only after the complete immutable commitment set closes and reuses it on retries. This changes capture throughput and fairness, not miner deadlines, submission validity, sample selection, grading or rewards.

`prospective_collection_timing.py` changes only new opening metadata at an explicit signed round boundary. It preserves the original fresh-run configuration, refuses to rewrite already-issued windows, accepts the exact prospective profile on restart, and keeps job lifetimes and scientific execution unchanged.
