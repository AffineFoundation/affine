# Verifier cache capacity and redundant replicas

Completed-submission retention removes authenticated local ZIP downloads only after canonical R2 size/SHA readback. It preserves job manifests, reports and failed-attempt logs. Obsolete-checkpoint retention preserves current and pending checkpoint identities. Neither policy alone prevents a new source workspace from downloading another physical copy of a protected checkpoint.

The E9 verifier-2 capacity incident demonstrated this distinction. Its completed-ZIP timer was working, but checkpoint retention had no active verifier cycle. The worker had no explicit mapping for current CP093, downloaded a separate copy in its new workspace, and ran out of disk while writing submission downloads. The partial downloads were cleaned; three preserved attempts for each affected request recorded ENOSPC. Root recovered those audits through the other verifier without penalizing miners.

The separate redundant-replica policy keeps identity protections unchanged. `ops.verifier_cache_capacity_plan.py` hashes both complete, explicitly named replicas, checks exact membership and stable device/inode metadata, and conservatively estimates physical reclaim. It never authorizes mutation. Existing hardlinks reclaim zero; a destination with unknown additional links yields no reclaim estimate.

`ops.verifier_redundant_cache_lifecycle.py` supports a narrowly approved one-shot operation. Before retirement it requires authenticated full streamed R2 readback, explicit root/reference approval, the original worker terminal, idle GPU, no FD/cwd/mmap references, unchanged duplicate hashes/inodes and sufficient prospective capacity. The retained keeper must have exactly two known ordinary canonical aliases. Both keeper paths and every current/pending protection survive. A fsynced operation-start record precedes rename; rename/unlink failure records uncertainty. Existing operation records forbid automatic repeat and require forensic review.

Root applied this policy to the single-link E9 CP093 duplicate, recovering **15,361,839,104 allocated bytes**. Actual post-operation free space was **15,834,230,784 bytes**. All ten canonical R2 objects had independently passed complete streamed size/SHA verification. The original two-link CP093 keeper and protected CP660 were preserved. The worker restart uses the same source, identity, authority, coordinator and workspace, adding only an explicit `--checkpoint-cache CHECKPOINT_ID=VERIFIED_KEEPER_PATH` mapping. Its separate supervisor records actual child wait/exit and never automatically restarts. Original failure logs and launch evidence remain intact.

## Deployment and automation limits

This recovery is an explicit operator operation. The redundant-cache policy is **not yet integrated into an automatic timer**, and the active controller configuration was not edited. Record the actual worker CLI mapping in future reviewed deployment configuration; do not assume the original configuration now carries it. Admit and map pending/new checkpoints separately before they can cause another duplicate download.

For each future workspace transition, authenticate the normal checkpoint descriptor, verify a complete same-host cache, reuse that admitted path, and enforce capacity for missing checkpoint bytes plus the signed artifact maximum and an explicit reserve before admitting work. A deterministic capacity refusal should quarantine the worker rather than consume artifact-verification retries. Integrate verifier cache maintenance with normal completion using current controller/source bindings and explicit scope records. Never clear global protections or infer an unreferenced cache from expiry alone.

Local controls:

```sh
.venv/bin/python -B -m unittest discover -s tests -p 'test_verifier*cache*py'
```

Fourteen controls cover actual copy/hardlink accounting, unknown links, corruption, symlinks including alias-directory roots, membership changes, exact protected IDs, GPU refusal, FD/cwd/mmap references, prior-operation refusal and uncertainty on rename/unlink failures. They do not qualify numerical model behavior or change the signed science contract.
