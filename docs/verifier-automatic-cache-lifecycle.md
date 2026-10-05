# Automatic verifier cache lifecycle

R2 owns durable checkpoints, submissions, and reports. Verifier disks hold disposable input copies. `CacheLifecycle` records a cheap filesystem receipt immediately after an existing authenticated SHA check; deleting that disposable input never requires another checkpoint hash or R2 download.

The worker leases its checkpoint while the backend and report transport are running. It passes the lease file descriptor to the backend, so a surviving child retains the lock after worker death. Completed prior checkpoints are evicted automatically before the next checkpoint hydrates. The current checkpoint remains reusable, bounded to one version per managed backend workspace. Fresh backend subprocess termination releases GPU tensors. Submission inputs are released after an authenticated coordinator report ACK; pending reports, job JSON, diagnostics, signing keys, source, and control history remain. Failed transport leaves inputs and pending reports intact. Successfully downloaded inputs can also be released after an authenticated job-failure ACK; no failure is reclassified as fraud.

Receipts bind the approved inventory and verified digest to device/inode/size/mtime/ctime/ownership. Eviction fails closed for active leases, changed files, extra unowned members, symlinks, hardlinks, or unmanaged directories. Successful partial member hydration can be retired later only if all actual members have receipts. Mapped external approved checkpoint caches remain untouched unless an operator explicitly adopts an exact owned root.

## Trainer API

After authority publication and durable readback ACK, while holding `lease_checkpoint(cp_id)`, call:

```python
cache.adopt_checkpoint(cp_id, export_path, approved_files, durability_ack)
cache.evict_checkpoints(exclude=[new_current_cp_id], keep=0)
cache.retire_downloads(original_job_id)
```

`export_path` is restricted to `root/checkpoints/<cp_id>` (explicit adoption also accepts historical `root/checkpoint/<cp_id>`) or `root/jobs/<job_id>/checkpoint-persistent-final` (also `checkpoint-final`). The caller must supply already authenticated model inventory and durability acknowledgement. The API does not create a new proof or check publication itself. Never retire the trainer successor before model and optimizer durability are acknowledged.

## Source-preserving deployment for historical jobs

Historical jobs pin the original source inventory. Do not overwrite original science6a `backend_jobs.py` or `distributed_worker.py` merely to add operator cleanup. Stage the immutable lifecycle module, worker overlay, and `ops/automatic_verifier_lifecycle_worker.py` in a separate namespace. Run the wrapper with `--backend-source <original-pinned-source>` and the existing worker arguments. Provide an explicit ROOT `--source-registry` JSON map keyed by the exact signed `manifest.source_bundle.sha256`, containing only operator-approved absolute `path` and exact `source_files`. The overlay routes old science6a and new sealed source variants to their corresponding installed backend directories, rejects unknown archive hashes or inventory mismatches, and verifies each registered runtime inventory once at worker startup/use. Miner manifests never choose filesystem paths. The wrapper uses original protocol/authentication modules and starts the original backend in its original source directory; backend job/source checks remain intact. Successful backend execution plus coordinator report ACK supplies existing verified inventory receipts to the overlay, without a second model hash.

For historical caches on quarantined full nodes, ROOT first confirms all original readers of the exact catalogue roots are stopped. ROOT signs an `owned-verifier-cache-catalog-v1` payload with finite created/expires timestamps, `quiescent_readers_confirmed`, and exact roots/checkpoint inventories/durable ACKs/keep IDs. `ops/adopt_owned_verifier_cache_catalog.py` verifies that signature and lifetime and uses the same lifecycle eviction API to recover disk automatically. No discovery, recursive rm, source/key cleanup, failed queue reset, or historical report deletion occurs. Read back actual free capacity, then qualify original backend on the retained/new checkpoint before enrolling the worker. For healthy workers, drain their exact job and stop the original process before bootstrap adoption; otherwise their old process would not hold lifecycle locks.

Activation requires ROOT review of the code, exact owned-root catalogue, original process quiescence, staged wrapper hash, original source hash, individual signing key, coordinator trust, and returned disk capacity. This document does not claim any production activation.
