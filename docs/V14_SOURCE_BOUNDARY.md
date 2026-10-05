# Prospective asynchronous mining and level-one transport boundary

This source update is preparation only. The approved a65 source and any active
epoch retain their original signed manifest, sampler, proof policy, deadlines,
compression and source binding. It does not claim an observed speedup, accepted
sample, training update or one-hour completion.

The next signed configuration selects
`artifact_compression_policy={"version":"lossless-deflate-v1","level":1}`
with existing per-pair small commitment transport and bounded hourly policy.
An absent compression policy retains historical level six. Both public and
owned miners follow the same exact policy; the optional CLI compression flag
only asserts it. Compressed/expanded byte limits and all tensor/proof bytes
remain unchanged. Existing acknowledged pair slots are not recompressed.

With the reviewed asynchronous dispatch code, one original owned job is
persisted, and collection proceeds at the original deadline while that job may
still run. Its report is not an audit or reward. The original supervisor/child
reservation prevents another job on the same physical miner until genuine
terminal observation; absent, ambiguous or unreachable handles fail closed.
This can prevent mining observation from consuming the freeze window, but does
not promise a fixed network, copying, audit or publication duration.

Before activation, finish the actual current epoch with its original sealed
source and original scientific handles. An original `--once` coordinator must
exit before opening another epoch; do not interrupt active audits or manufacture
an idle state. Stage new code in distinct directories first. Then require
`controller.active == None`, no outstanding scientific claims, the actual
learned checkpoint, and exact equality of controller and durable trainer-state
journals. Carry the observed optimizer counter and parent, including any new
training update; never hard-code a prior checkpoint, round or genesis.

ROOT alone authorizes the ordered successor source grant and parent continuity.
Preserve all historical grants and the same queue/state, verifier identities,
scoped seeds, caches and forwards. Start one successor coordinator, publish its
actual PID/ticks to discovery, preserve the single historical-compatible reward
writer, and replace evaluators only when idle. The first genuine next-source
mine/queue/audit/training receipts establish deployment validation.
