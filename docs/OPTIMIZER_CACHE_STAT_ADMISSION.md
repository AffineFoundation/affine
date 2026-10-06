# Prospective owned optimizer cache admission

This default-off source candidate introduces the explicit policy:

```json
{"version":"sole-current-fp32-state-cache-stat-v2","validation":"durable-unchanged-inode-v1","max_checkpoint_bytes":137438953472}
```

Historical v1 and absent policies retain full SHA verification. This candidate is not deployed and does not qualify the running source0380 job retrospectively.

A v2 promotion still hashes every retained shard in full, authenticates the original ROOT durability ACK and its exact signed job/source/descriptor, and saves a private verification stamp over that catalogue. A missing stamp requires full hashing. Future admission requires the identical descriptor, checkpoint, counter, source and 23-file inventory plus the unchanged device/inode/size/mtime/ctime/mode/uid/single-link identity, under the exclusive existing optimizer lease. Missing or changed owned state falls back cold or refuses unsafe paths; foreign state is never adopted.

Each admitted shard is opened with NOFOLLOW before its owned rename. A private rename journal binds the exact descriptor and ACK, both stat identities and the destination. Only the rename's ctime may differ. Restore uses the same opened file descriptor through `/proc/self/fd`, validates its identity and the journal before and after materialization, and closes it even on failure. Tensor keys, FP32 dtype, shapes, finite values, nonnegative second moments and disjoint copy checks still execute. This eliminates the second SHA pass only on that explicitly authorized path. Network fallback remains fully hashed. Receipts label prior full SHA verification and explicitly state that the current restore did not hash again.

The private trainer filesystem, ownership catalogue and qualified lifecycle code are trusted. Stat identity is not a cryptographic defense against a malicious privileged host rewriting both bytes and its catalogue. ROOT-signed durable parent state remains the authority; no ROOT signature is fabricated for a local stat observation. Source changes do not alias an old cache into a new source.

New timings include `parent_cache_validation_and_admission` and `parent_cache_and_restore_total` in both transport diagnostics and startup phases. Existing `parent_state_restore` stays a separate phase, so the earlier serial preparation cost is visible.

Before production, qualify the new source pins, signed policy, full 23-file promotion and reuse on an idle qualified trainer. Compare exact restored buffers and the next optimizer update against the full-hash baseline; test real owned rename/lease continuity and all cold fallback paths. CPU fixture measurements establish only the local mechanism, not a 91GB production speed claim. Independent full-state physical readback, export SHA, optimizer lineage, numerical parameters and single-writer publication gates remain unchanged.
