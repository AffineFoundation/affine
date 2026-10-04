# Recover original training before checking a new allocation

A completed training job can consume the disk reserve that allowed it to start.
Requiring that entire reserve again before retrieving its report can leave a
successful epoch stuck in training. The controller now checks for an original
signed request before applying the reserve for a new job.

Resume binds the exact manifest, steps, fixed training policy, ordered frozen
submission hashes and replay inputs to the original signed job and dispatch
record. An existing checked report or an authoritative running/complete status
reuses that job. Changed inputs, failed or absent original jobs refuse; they do
not silently launch another training attempt. New jobs still undergo the full
checkpoint/output/download/decoding reserve check.

Publishing an existing checkpoint streams its files and reserves one MiB for
job metadata, rather than another pair of checkpoint copies. The upload worker
still hashes the complete pinned files, and the operator independently reads
every R2 object before publishing the immutable checkpoint descriptor. No
model/proof tolerance, audit policy or optimizer objective changes.

During the October 4 epoch-six recovery, the operator authenticated the original
completed three-update report and protected its actual successor in controller
state, retaining the active training phase and nine cumulative completed updates.
Publication and metrics recovery reused that completed output without rerunning
training or changing the signed epoch source. All ten R2 objects, totalling
15,242,726,234 bytes, passed independent full-byte hash readback before publication.
The original controller advanced to post-training evaluation. Public code fixes
apply to a future admitted source. Original job evidence remains private.

The obsolete-export guard now also checks the original current-epoch training
job/report and protects its completed successor before `next_checkpoint` appears
in controller state. A failed original housekeeping worker exited one and was
preserved; the corrected worker is deployed with a new original process record.
Neither cleanup worker deletes the original job/report or the protected successor.

Seven controls exercise disk-pressure recovery, original live-job reuse,
changed request/signature/report refusal, failed/missing-job refusal, bounded
streaming-upload space and controller recovery before new-job capacity checks.

The same older-source reserve problem recurred after the original epoch-seven
training finished at 07:18 UTC on October 4. Root authenticated its original
signed job, successful runner wait and process absence, source/runtime bindings
and all successor weight bytes. Scoped recovery collected that existing report
and streamed its successor to R2; all ten files (15,242,726,234 bytes) passed
independent full-byte readback at 07:40 UTC. No training was repeated and no signed
epoch source/deadline changed. The original controller advanced to its original
post-training evaluation. The admitted future source already includes recovery
before a new allocation check; it remains pending production activation.
