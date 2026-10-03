# Bounded frozen submission publication

The October 3 live epoch froze 33 in-window uploads totaling about 17 GB.
Its pinned controller reads and publishes them serially before issuing audits.
The controller continued making byte-level progress and eventually published all
33; it was not restarted or given a later deadline. This revealed a throughput
and memory problem worth addressing before raising the three-batch pilot cap.

The prospective gateway uses four publication workers, configurable from one
to eight. Each worker reads an operator-owned immutable frozen object in 1 MiB
chunks, checks its complete SHA256 and exact size against the original receipt,
closes the stream, and makes a server-side copy to the public submission key.
Copying is conditional on the ETag returned by that same read, so an intervening
replacement fails instead of publishing different bytes. ETags do not replace
SHA256 checks. Cloudflare documents support for source-conditional copying in
its [R2 S3 API compatibility reference](https://developers.cloudflare.com/r2/api/s3/api/).

Publication never copies the miner's mutable staging key. Initial atomic GET
snapshots, original completion times, deadlines, private frozen hashes and
rejection rules remain unchanged. A failed copy or integrity check leaves the
final receipt set unset. Retry rechecks the existing frozen objects; it cannot
resnapshot a later miner replacement. Partial public copies are harmless
immutable outputs and do not independently become accepted audit evidence.
The final receipt set is published only after every selected frozen object has
passed its check and copy. Initial snapshot collection is still serial and can
still buffer an entire submission; this change does not solve that separate cost.

Eight focused controls cover bounded stream reads, corruption, changed size,
truncation, the read/copy race, concurrency limits and retry behavior. Eight
existing direct-R2 controls and three gateway protocol controls also pass.
An actual private R2 probe used four 8 MiB objects: serial checked copying took
3.55 seconds and four-worker copying took 0.93 seconds. Independent full-body
readback matched all eight destinations; a deliberately replaced source caused
the real conditional copy to return HTTP 412. This small probe is not a
throughput guarantee for 17 GB epochs or a production rollout claim.

The prospective source must be admitted on the controller and role workers and
approved for a future epoch before activation. Existing signed epochs keep their
original code, source anchors and audit history.
