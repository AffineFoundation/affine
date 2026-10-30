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

## 2026-10-30 14:16 CR follow-ups
- `activate_f.sh`: re-activates v8.3.3f (fp32 master) after a false-positive fp32v auto-revert. It uses the sanctioned fp32v drop-in and switches only at a safe boundary:
  - no open TRAINING-JOB without a TRAINING-RESULT in epochs C..C+2 (CURRENT lags pre-opened epochs);
  - no RECOVERY-REQUIRED;
  - lifecycle lock + `LIFECYCLE-ACTIVATION.lock` held;
  - pod free >= 192 GB (135.1 GB job disk check + 36 GB fp32 entry + 20 GB margin).

  Revert: `MODE=revert bash activate_f.sh`.
- `watchdog_ts.py` counts only trainer-caused faults. It does not count a fault when any of these hold:
  - the pod is short of disk;
  - the job text mentions disk/space/capacity;
  - our worker never started the job.
- `fp32_prune.py` (service `ts-fp32-prune`) keeps only the king's fp32 entry plus the in-flight candidate's. After each gate decision it deletes the other entries: on RED the candidate's entry, on GREEN the old head's. Entries newer than the decision and PINned entries (upload in progress) are kept. Revert: `systemctl --user stop ts-fp32-prune`.
- `activate_ts.sh` (v8.3.3g) is now based on live v8.3.3f. It waits for one clean v8.3.3f epoch (`AFTER_OK`), runs the same disk precheck, and stops `ts-watchdog-f` before switching.
