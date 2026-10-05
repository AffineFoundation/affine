# Eight-stream transport candidate

This preparation adds exactly one separately signed transport opt-in:

```json
{"optimizer_state_transport":{"version":"bounded-eight-fp32-state-v1","concurrency":8}}
```

An absent field still selects one stream. The existing
`bounded-parallel-fp32-state-v1` version still admits only integers 1 through 4.
The new version admits exactly integer 8. Values 5 through 7, booleans, floats,
unknown versions and unsigned or mismatched resource admissions remain rejected.
No optimizer arithmetic, dtype, tolerance, shard cap, tensor allowlist, step
counter or authority publication rule changes. Export and restore use the same
bounded worker pool and complete SHA, size and tensor checks.

Production-sized inventory arithmetic for eight streams reserves 91,387,398,144
FP32 state bytes, 32,000,000,000 bytes for in-flight shards, 145,057,269,760
additional RAM bytes including the existing largest-parameter and 8 GiB reserve,
and 55,832,660,826 additional disk bytes including the BF16 export. A CPU control
also keeps 15,231,233,024 BF16 parameter bytes, requiring 160,288,502,784 total RAM.
Actual idle trainer measurement at 1791173833 had 187,816,951,808 conservative
cgroup-bounded usable RAM bytes and 1,221,905,289,216 free disk bytes. This fits by
27,528,449,024 RAM bytes at that observation; fresh resource admission remains
mandatory. Its 16 CPU cores can make materialization contention different from
retained e566's 128 cores. Resource fit does not prove throughput or GPU capacity.

The four-stream production-sized control completed 23 actual PUT/full GET
receipts totaling 91,387,491,264 bytes at 1791174593.1983. Its original supervisor,
root request, private namespace and ongoing restore are untouched. The next
CPU-only eight-stream control must wait for that original actual terminal and
for the dedicated reader qualification measurement. Use a new sealed source
with this one transport extension, a new root-reviewed signed helper/request,
the same approved CP28 full ten-file map and parameter inventory, and a separate
private namespace. Explicit isolated genesis 0→1 with one zero-gradient CPU
update is qualification only. Preserve all failed and partial histories.

The next control must record all 23 real PUT/full GET receipts, exact restored
FP32 buffer comparisons, complete phase times, actual maximum eight in-flight
shards, memory/disk admission, and original PID/start ticks/actual wait terminal.
Compare the resulting per-shard bytes and hashes against the four-stream control
for the same zero-gradient state where complete checkpoint serialization identity
matches. Preserve unchanged original CP hashes and all flags forbidding authority
promotion. No source activation or hour claim follows from this preparation.
