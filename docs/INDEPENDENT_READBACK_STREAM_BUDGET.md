# Prospective independent optimizer readback streams

This implementation is review-ready and has not been deployed. Existing signed
requests without `stream_budget` retain exactly four workers and the original
receipt shape/concurrency. Original E15 source, requests, processes and lineage
remain unchanged.

A future signed epoch manifest can explicitly carry:

```json
{"independent_state_readback_budget":{"version":"bounded-independent-readback-streams-v1","concurrency":8,"ram_reserve_bytes":1073741824}}
```

The reader configuration must contain the identical `stream_budget`, and its
ROOT-signed actual qualification admission must contain that exact budget in
addition to the original module hashes, physical reader identity and full
23-object integrity qualification. Configuration cannot upgrade a historical
qualification. Four and eight are the only accepted explicit counts; booleans,
floats, other counts, unknown fields and weak RAM reserves are rejected.

The coordinator inserts this budget into the original ROOT-signed readback
request. Authority commitment binds it back to the exact original signed
training manifest; an eight-stream request cannot be grafted onto a historical
four-stream manifest, and a four-stream receipt cannot be relabelled as eight.
Every original object still receives a complete byte-length and SHA256 check.
Descriptor/tensor metadata, source, original job/report, independent physical
machine, original child terminal, deadline, optimizer counters and authority-last
publication rules are retained.

Both coordinator preflight and the actual reader execution admit available CPU
RAM before any object read. The conservative allowance is 8 MiB per streaming
lane plus a signed reserve of at least 1 GiB (at most 64 GiB). Eight lanes thus
require at least 1,140,850,688 bytes available, and do not hydrate 91 GB of state
into local disk or RAM. Admission reads Linux MemAvailable and caps it by visible
cgroup-v2 memory.max minus memory.current when finite. This is measured resource
admission, not OS resource enforcement or a throughput guarantee. Chunk size is
still 1 MiB and maximum object size 4 GB. No GPU, model inference or training runs
on this reader.

Deployment requires a NEW isolated reader module namespace, pinned updated module
hashes/helper/supervisor, actual full 23-object eight-stream transport controls,
a new ROOT-signed qualification and a prospective signed manifest. Do not overwrite
modules used by a live original readback. The old namespace must remain available
for historical recovery and its old module-byte checks. E16 native-only source c1
was prepared earlier; this change belongs in a newly admitted source variant or
an explicitly reviewed controller reader overlay, not an in-place archive edit.

## Original role startup timing

Prospective worker reports now include `startup_timings` with completed original
operation counts and monotonic elapsed seconds for job validation, source/runtime
authentication, task asset hydration, checkpoint materialization/authentication,
runtime/model construction, cumulative input download/authentication and eligibility
admission, native prompt eligibility, and parameter digests before/after training.
They do not run any new model verification. Failed operations propagate normally
and are not represented as completed-success timings. These are wall times without
new GPU synchronization; existing synchronized training diagnostics remain the
source for GPU phase measurements. Runtime construction can include constructor
hashes/model load and other setup, so subdivide only after observing the report.

This exposes the previously uninstrumented roughly 15-minute E15 pre-restore
interval, rather than attributing it all to native environment construction.
The prospective instrumentation changes backend_jobs source bytes and therefore
needs a separately signed source bundle/qualification before production use.
