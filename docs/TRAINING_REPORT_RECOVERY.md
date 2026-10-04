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
Publication and metrics recovery reuse that completed output; they do not rerun
training or change the signed epoch source. Public code fixes apply to a future
admitted source. Recovery progress and original job evidence remain private.

The obsolete-export guard now also checks the original current-epoch training
job/report and protects its completed successor before `next_checkpoint` appears
in controller state. A failed original housekeeping worker exited one and was
preserved; the corrected worker is deployed with a new original process record.
Neither cleanup worker deletes the original job/report or the protected successor.

Seven controls exercise disk-pressure recovery, original live-job reuse,
changed request/signature/report refusal, failed/missing-job refusal, bounded
streaming-upload space and controller recovery before new-job capacity checks.
