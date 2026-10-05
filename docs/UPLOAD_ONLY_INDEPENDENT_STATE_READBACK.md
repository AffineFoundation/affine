# Prospective upload-only optimizer export

The signed `optimizer_state_export_policy: upload-only-independent-full-v1`
requires `persistent_publication_policy.state_readback: qualified-remote-full`.
Without this field the trainer retains its complete shard PUT plus full GET
readback. Historical receipts and four/eight-stream controls remain unchanged.

For this prospective policy the trainer hashes every local FP32 shard, completes
its PUT, and retires the bounded local transfer file. It still PUTs and fully
reads back the small staged descriptor. Shard receipts explicitly report
`durable_readback_verified: false`, `export_verification:
"uploaded-local-sha-only"`, and `independent_full_readback_required: true`.
These receipts are insufficient to publish weights or authorize rewards.

An admitted reader on a different physical machine must fully GET and hash
every shard. The operator authenticates the original signed request, exact
object inventory, source/model/optimizer lineage, reader identity, and original
successful terminal evidence. There is no local or unqualified fallback.
Checkpoint PUT and full GET/hash staging may overlap this reader, but neither
checkpoint nor optimizer authority descriptor is signed until both validations
pass. Only then does the coordinator commit the checkpoint and state and issue
the durable reward-readiness artifact. Empty/no-update epochs remain distinct.

The optimization removes the trainer's duplicate full output-shard GET. It does
not skip the sole mandatory independent full readback, parent state restore,
proof verification, parameter updates, checkpoint hashing, or lineage checks.
Real production-sized timing and ROOT source admission remain deployment gates;
no measured one-hour completion is claimed by these CPU controls.
