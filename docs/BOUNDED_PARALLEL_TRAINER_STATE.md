# Bounded parallel optimizer state transport

An absent `optimizer_state_transport` field preserves serial worker state export.
A prospective signed epoch can specify:

```json
{"optimizer_state_transport":{"version":"bounded-parallel-fp32-state-v1","concurrency":4}}
```

Only integer concurrency 1 through 4 is accepted. The signed manifest selects
both worker execution and the coordinator's capacity requirement; the worker
repeats actual admission after loading its model. Each admitted lane reserves one
complete maximum transfer shard in CPU serialization/readback RAM and local disk.
Four lanes therefore add 12 GB RAM and 12 GB disk reserve versus a serial 4 GB
shard plan. This is additional to all FP32 masters and moments, parameter temporary
reserves, the complete BF16 export, input/artifact reserves and safety margins.

The optimizer's publication freeze remains held until every worker completes.
At most four tasks materialize, SHA/size-check, PUT, and independently stream-read
back their own shard. Completed files are retired only after the actual entire
object matches. Shard metadata is collected in the original deterministic order;
the descriptor's bytes, optimizer buffers, AdamW math, counter and lineage do not
change. Only after every shard passes is the original descriptor-last callback
allowed to run. A partial failure prevents descriptor publication, cancels pending
work, waits for active transfers under the freeze, and retains failed shard files
and small forensic receipts. No partial state is admitted for restoration.

Reports include actual maximum tasks/transfers in flight and per-shard transport
start/completion timestamps. These are measured execution evidence, not a claimed
GPU-fleet speedup or a one-hour epoch guarantee. The coordinator already performs
its own bounded parallel readback before authority signing; this worker change
removes its serial staging bottleneck without skipping either readback layer.

Qualification uses tiny real BF16 parameters and FP32 masters/moments with actual
safetensors files: overlapping four-way transfers, exact serial/parallel descriptor
identity, deterministic order, round-trip restoration, corrupted readback rejection,
failed-file retention, and increased coordinator/worker resource bounds. Production
throughput must be measured after deploying an admitted prospective source.

The same signed prospective concurrency and resource admission also bound parent
restoration. Descriptor validation establishes exact disjoint tensor coverage
before any buffers or workers are created. Up to four workers download, hash,
open safetensors, check dtype/shape/finiteness and copy only their approved
non-overlapping slices. Evidence remains in deterministic original shard order.
No optimizer, training update or partially restored state is returned until every
shard passes. A failure cancels pending work and waits for every active worker;
failed data and forensic receipts remain, while independently verified successful
temporary files are retired. Absent prospective concurrency, restoration remains
serial. Actual restore overlap and per-shard timestamps are recorded separately
from the model's optimizer counter.
