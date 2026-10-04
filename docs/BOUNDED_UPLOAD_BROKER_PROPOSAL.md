# Prospective bounded upload transport

Status: **implemented as a local primitive, tested with synthetic data, and
size/conditional headers qualified with a tiny real R2 object; not integrated
into a miner, controller, active epoch, or deployed service.**
The live raw-PUT transport is unchanged. This proposal changes transport, not
the sampling proof, batch scoring, or permitted task population.

## Purpose and provider constraints

Rejecting an oversized object at freeze protects verifier resources after the
object has already arrived. A bounded upload grant instead negotiates the exact
compressed archive size before upload. Cloudflare documents reusable presigned
PUT URLs and does not support S3 POST form uploads, so a POST
`content-length-range` policy is unavailable here. [R2 presigned URLs](https://developers.cloudflare.com/r2/api/s3/presigned-urls/)

The prospective immutable mode additionally signs `If-None-Match: *`, which R2
lists as supported for PutObject. Actual cloud enforcement still requires an
operator qualification run before admission. [R2 S3 compatibility](https://developers.cloudflare.com/r2/api/s3/api/)

## Primitive and authority boundary

`subnet/upload_broker.py` exposes `SizeBoundUploadBroker(database,
authority_key, bucket)`. The operator supplies the existing authority signing
key and an existing bucket client; neither is given to miners. The SQLite file
is private and must reside on a durable operator volume, outside disposable
machine-cache retention. Losing it loses issuance accounting, so recovery must
restore it or close the affected epoch rather than create a fresh empty ledger.

`register_epoch(signed_policy, signed_registration_snapshot)` authenticates both
documents and their exact binding. The registration snapshot must already have
passed the controller's chain activation checks. The broker authenticates that
snapshot; it does not query chain registration independently. Policy is immutable
per epoch and contains:

- Original epoch start/deadline and sorted activated Ed25519 identities.
- SHA-256 of the complete authority-signed registration document.
- Maximum compressed bytes per snapshot, maximum grants per identity, and
  aggregate reserved declared bytes per identity.
- Grant lifetime at most 300 seconds and minimum interval between new grants.
- Explicit `immutable-snapshot` or compatibility `cumulative-replace` mode.

The wire constants are `bounded-size-upload-broker-v1` and
`direct-r2-size-bound-v2`. The hard compressed-size ceiling remains
2,000,000,000 bytes; an epoch can set a smaller ceiling. Grant count is bounded
at 256 per identity. These are validation ceilings, not recommended launch
budgets. Existing batch, rollout, decompression, and proof-work limits still
apply independently.

## Request and cumulative miner uploads

A miner constructs the cumulative archive locally and measures its exact byte
length, including all occupied batch slots. It requests a grant using either:

1. An activated Ed25519 signature over the exact grant request; or
2. An operator-signed, epoch/identity/policy/deadline-scoped delegated bearer
   ticket delivered privately or encrypted to that identity. This permits an
   operator's remote test miner without exporting a wallet private key.

The signed request payload has exactly `version`, `action: "grant"`, `epoch`,
`identity`, a fresh 64-character hexadecimal `nonce`, integer `bytes`, and
finite timestamp `at`. The response is an authority-signed document binding the
policy, identity, nonce, exact object key, original deadline, actual signature
expiry, exact byte count, required request headers, and remaining issuance
budgets. Neither requests, bearer tickets, nor response URLs belong in public
logs or manifests. Requests and responses use `Cache-Control: no-store`.

For immutable mode, each nonce names a distinct object:

```text
private/{epoch}/staging/{identity}/{nonce}.zip
```

PUT signs `Content-Length`, `Content-Type: application/octet-stream`,
`If-None-Match: *`, and the normal host header. The miner verifies the authority
signature and all fields, then sends the exact archive bytes with those headers
without chunked transfer or intermediary transformation. A changed archive size
requires a new grant. The cumulative file contents stay unchanged in meaning;
only the staging object key changes for each snapshot.

SQLite transactions atomically enforce rate, grant count, and the sum of all
declared sizes. There are no refunds during an epoch. An exact request retry
with the same nonce returns the same original grant without spending another
grant; a conflicting payload refuses. Once that grant expires, it cannot be
renewed with the old nonce. This is issuance idempotence, not evidence an upload
completed. A failed conditional PUT must not be treated blindly as success:
retry recovery must confirm the operator-observed object bytes/digest match the
archive, or abandon the snapshot and request a new bounded grant.

Compatibility mode uses the previous one-object-per-identity staging key and
can overwrite it repeatedly. It is explicit and has weaker storage-write cost
bounds; grant issuance limits do not limit successful reusable-URL overwrites.

## Freeze and historical uploads

After closure, `finalization_candidates(epoch, identity)` returns only known
issued keys and their size/window metadata, without URLs or secrets. The future
freezer must HEAD those known keys and choose the latest **actually completed,
in-window** snapshot using bucket metadata, never a miner-supplied timestamp or
an unrestricted listing. Tie-breaking must be deterministic. Late or unfinished
uploads do not qualify merely because their grant was issued before closure.

Freezing must independently check actual ContentLength equals declared bytes
and is within epoch limits, stream/hash the complete object, and use the current
ETag/conditional-copy integrity boundary before publishing a frozen artifact.
Archive framing, decompression, batch counts, proofs, and outcomes remain
checked separately. A malformed final snapshot does not automatically revive
an earlier one; that fallback must be an explicit public epoch policy if desired.

The request timestamp is accepted only within 60 seconds of broker time, even
for an identical nonce retry. A client should persist the returned signed grant
locally and retry promptly; a long reconnect can require a fresh nonce and spend
another issuance reservation while the old URL remains valid. Extending retry
recovery to the complete original grant lifetime would be a prospective API
change and needs a test before claiming that behavior.

Historical raw-PUT grants remain valid until their original expiry. This broker
cannot revoke or retroactively meter them. Adopt v2 only through a newly signed
epoch and admitted source inventory, with the matching public miner and
`/llms.txt` contract. Never mix known v2 snapshot keys with historical same-key
freeze rules implicitly.

## Hosting and required qualification

The optional `UploadBrokerServer` binds loopback and exposes
`POST /v1/upload-grants`. It bounds request framing to 16 KiB, refuses chunked
requests, applies a five-second socket timeout, suppresses access logs, and
returns generic refusal bodies. The existing HTTPS gateway must proxy this route
with explicit connection/request-rate limits. The Python thread-per-request
server alone is not a public load-management boundary. Policy registration,
delegation issuance, and finalization remain operator-only Python APIs.

Before integration, the operator must qualify a tiny real object against the
actual endpoint, using synthetic bytes and disposable staging keys: confirm
matching signed length and condition succeeds, mismatched length or required
headers/key refuses, replay cannot overwrite an existing immutable object, and
expiry/window handling matches freeze semantics. Local offline signing proves
the headers appear in SigV4; it does not prove the provider rejects a mismatched
PUT. On October 4 the operator qualified a 17-byte synthetic object against the
actual bucket: matching signed length/condition returned 200; reuse could not
overwrite it (412); a changed body length, omitted signed conditional header,
and expired original URL each returned 403. Full GET readback matched all 17
bytes and their SHA-256. Records are in
`state/root-audits/tiny-size-bound-r2-qualification-20261004-v1`.
This qualifies those provider header checks, not the end-to-end broker/client,
HTTPS hosting, epoch integration, or large-file throughput. No active epoch or
historical grant changed, and the synthetic object remains in qualification
storage.

Keep the durable SQLite ledger and staging records until epoch finalization and
its audit record are persisted. If staging retention is later configured, scope
it narrowly and preserve finalized evidence/checkpoints. Lifecycle expiration
is eventual cleanup, typically within 24 hours, rather than immediate admission
control. [R2 object lifecycles](https://developers.cloudflare.com/r2/buckets/object-lifecycles/)

## Limits and local validation

Immutable signed keys bound successful retained snapshots and total reserved
bytes, conditional on R2 enforcement. They do not remove request-processing
cost, repeated failed requests, multiple registered identities, a compromised
operator, or infrastructure outages. Do not assume every failed conditional
request is free or that URLs can be individually revoked before their short
expiry. Storage/request abuse is separate from token or sampler verification.

`tests/test_upload_broker.py` passed 12 synthetic local tests covering signatures,
scope, size/header binding, nonce retries, rate/byte quotas, restart persistence,
atomic concurrent issuance, metadata-only finalization, safe local HTTP framing,
and real offline SigV4 header construction using explicitly fake credentials.
The only HTTP exercise is an in-process localhost test; no external requests,
real credentials, object uploads, deployments, or active epoch changes were made.

```sh
.venv/bin/python -m unittest discover -s tests -p test_upload_broker.py -v
```
