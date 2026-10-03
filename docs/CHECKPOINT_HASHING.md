# Complete checkpoint hashing with bounded readers

`subnet.model.model_files` still reads every model-relevant file in full on every
invocation. It uses at most four concurrent readers, each with a 1 MiB buffer.
It does not trust timestamps, ETags, filenames or cached digests instead of
checking bytes. Unexpected model files remain in the inventory, and a failed
read fails the whole operation. Existing callers still require exact equality
with the approved checkpoint file map before inference or verification.

On an idle retained extra machine, complete checks of the same 15,242,788,091-byte
checkpoint took 55.74 and 60.60 seconds serially versus 13.36 and 14.70 seconds
with four readers, in serial/parallel/parallel/serial order. Every run matched
all original hashes. A further CPU-only execution of the exact new hashing
functions checked those same original weights in 12.41 seconds. This establishes
an improvement in hashing on that machine, not whole-verifier throughput or
performance on every provider disk. No model inference ran for these measurements.

Six controls cover complete multi-megabyte shards, weight mutation, unexpected
model files, existing asset membership, disappearing files and empty/single-file
inventories. The wider checkpoint controls and runtime factory checks pass.

This implementation is prospective for a new signed source. Already installed
live epochs and the precommitted independent benchmark retain their original
source bytes. Admit a new source before deployment; do not copy this module into
an already signed run. Numerical and TOPLOC policies remain unchanged.
