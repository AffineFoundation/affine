# Prospective token-only learner transport

Only a newly signed `small-commitment-pairs-v2` epoch uses this format. Historical
v1 uploads and audited-only training retain their original contracts.

Each signed child retains the proof ZIP hash, size, task index and canonical
batch hash. It also binds `training_sha256` and `training_size` to a separate
canonical JSON document, limited to 2,000,000 bytes, with exactly
`version`, `epoch`, `checkpoint`, `miner`, `slot`, and `batch`. Its version is
`committed-training-documents-v1`. Both owned and public miners upload proof ZIP,
token document, then the cumulative signed commitment. Acknowledged slots are
append-only across restart. Separate per-slot presigned token URLs convey no
bucket credentials.

For explicit `committed-unaudited-training-v1`, the operator captures all small
signed commitments, downloads only the bounded token documents, verifies their
complete byte hashes and scope, and journals immutable snapshots. Learner input
is explicitly unaudited. Declared proof ZIPs are neither fetched nor described
as verified during this capture; independent audit workers later capture and
verify selected proofs under their separately signed policy. Missing or malformed
token documents structurally exclude their miner without a fraud finding;
ambiguous infrastructure failures retry until the signed capture cutoff. At that
cutoff, only genuinely captured documents enter the learner; remaining slots are
explicitly infrastructure-deferred without fraud or invented verification. The
full small signed commitment population remains available to the independent
auditor.

These are admission byte limits. Presigned object PUT does not itself enforce an
ingress byte quota; oversized private uploads remain an explicit storage-abuse
limitation until an enforcing upload gateway is deployed.
