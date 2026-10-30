# SN120 trainer speedups: host v8.3.3g (RA/20261030-trainer-speed)

v8.3.3g is the live v8.3.3f trainer host with execution-only speedups. Only `mimo_host_trainer_v83.py` and `v83_dp.py` change. The engine, the original-method pins and the source root are the same, and the release is ROOT-signed with `sign_v833g.py`. A CPU test (`sanity_v833g.py`, v833f vs v833g, 3 fresh-adam fp32-master steps covering the bootstrap and memory paths) gives bit-identical params, masters and digests.

| change | what it removes (py-spy, e275 cold run, 710 s job->RESULT) |
|---|---|
| threaded per-tensor sha256 digests (48 threads, largest first); bf16 + fp32 digests computed concurrently | `_sha_tensor` ~130 thread-s on one core |
| fp32 master step: GPU `isfinite` + direct `masters[n].copy_(w32)`; torch threads 32 | CPU isfinite + extra CPU copy ~59 s |
| next `_fp32_prepare` reuses the digest from the previous finish (memory) / the verified meta digest (disk) | ~17 s fingerprint per job |
| fp32 master save (36 GB) deferred until after R2 publish (max 600 s) | contention with shard hashing / STAGED |
| `file_hash` memoized by (path, dev, ino, size, mtime_ns, ctime_ns) + thread-pool prefetch of sibling shards | file_hash ~88 s (helpers 31.6 s) |
| helper ranks preload KL-ref + actor before the first job from a hint written by rank 0 (hash-verified, falls back to lazy) | ~69 s first-job helper load in the update |

Activation: `MODE=switch [WAIT_DECISION=E] bash activate_ts.sh`. Revert (one line): `MODE=revert bash activate_ts.sh`.
`watchdog_ts.py` auto-reverts on any of these, during the first 3 v833g trainings:
- training > 13 min;
- TRAINING-RESULT not successful;
- worker dead with no restore running;
- a gate integrity reason.

`v833f-to-v833g.diff` is the diff against the live v8.3.3f host files. The private ROOT-ADMITTED config and the signing seed are not in this repo.
