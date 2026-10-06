# Measured collection and learning throughput

Read-only evidence from completed E28
`nonpayable-live-reward-math-v1--1791299272-28`, captured 2026-10-06.
No live manifest, deadline, optimizer, remote job or historical classification
was changed by this diagnosis.

## Actual population losses

| Stage | Count |
| --- | ---: |
| Miners with signed commitments | 141 |
| Committed batches | 403 |
| Captured token documents | 281 |
| Infrastructure-deferred document slots | 122 |
| Duplicate-task document exclusions | 68 |
| Eligible documents / training tasks / gradient pairs | 213 |
| Training document cap | 256 |

128 of 141 miners committed all three allowed batches; seven committed one and
six committed two. Thus 403/423 available commitment slots were filled. The
trainer cap did not discard E28 eligible inputs. Historical duplicate-zero
admission intentionally excludes every document for an overlapping task;
changing this would change the scientific/admission contract and is not included.

E24–E28 deferred respectively 145, 78, 159, 96 and 122 document slots. Their
eligible populations were 309, 273, 253, 227 and 213; selected populations were
256, 256, 253, 227 and 213. Capture loss is recurrent, separate from the cap.

E28's signed deadline was 1791300103 and freeze cutoff 1791300163. Commitment
metadata capture completed at 1791300119.655, leaving about 43.345 seconds for
token capture. Token receipts span 1791300119.949 to 1791300162.939, reaching
that bound. The 122 remaining slots carry `infrastructure_deferred`. Capture
previously rewrote the entire historical gateway snapshot after each receipt.
The current snapshot is 18,862,790 bytes; five CPU-only serializations measured
median 0.1444 seconds, equivalent to about 40.6 seconds for 281 serializations
before filesystem writes. This explains substantial bounded-window loss; it
does not individually attribute network versus filesystem delay for each slot.

## Prospective fix

`token_capture_journal.py` records a small private HMAC-authenticated fsynced
receipt after each successful immutable R2 publication. Capture replays receipts
against the exact original epoch, checkpoint, commitment child, slot, hash,
size, immutable key and original cutoff. A partial final append is discarded;
complete conflicting/tampered records fail closed. The whole gateway is written
at terminal projection rather than for every successful receipt. Existing
four-worker bounds, immutable bytes and deadline semantics remain intact.

First slots are scheduled across all miners before second and third slots;
epoch-bound hashed miner ordering avoids persistent low-identity preference.
This transport scheduling change does not change task selection or generation.
281 small fsynced receipt appends took 0.415 seconds on this local filesystem.
This is a CPU-only benchmark, not a completed live-epoch improvement claim.

Eight new controls cover restart without source rereads, crash after receipt
fsync, partial tails, tampering/changed commitment, expired capture, ownership,
and slot fairness. Existing training-document and training-input controls retain
original duplicate-task and structural eligibility behavior. No old deferred
slot is relabeled or old deadline extended. Carry this into a future immutable
source after immediate R29 calibration recovery; it need not block that recovery.

## Generation and objective findings

Selected E28 rollouts contain exactly 213 positive and 213 negative trajectories.
122 negative trajectories (57.3%) have exactly the signed harness limit of
1,024 output tokens; none of the positives reaches it. Positive outputs average
350 tokens (median 331), negatives 726 (median 1,024). This is evidence consistent
with many token-budget failures; terminal reason is not recorded, so exact
length alone does not prove truncation for every one. Environment spec allows
2,048, but the signed selected harness is 1,024. A token-budget change needs its
own controlled comparison and cannot be applied to old inputs.

The immutable training reference is recomputed from E28's input checkpoint
703a7a31..., and all 213 recorded pre-update margins equal their reference.
The first preference loss is therefore expected to be ln(2), independent of
the absolute before-margin. The objective uses output-token-mean logprob and
`-logsigmoid(0.1 * (margin - reference))`, averages pairs within each task, then
tasks within the update. One optimizer update was performed with persistent
FP32 master/AdamW state, learning rate 1e-5 and 339 gradient tensors. Recorded
gradient norm before clipping was 0.0381. There is no stale-reference evidence
in these reports. Mean training-margin delta was +0.0181; 187 pairs increased
and 26 decreased. Training loss and in-sample margin do not establish held-out
convergence; independent same-128 scores for CP11–18 were 77, 86, 84, 78, 85,
87, 83 and 86.

Aggregate attempts on exhausted tasks and unselected generated trajectories
are not available in the examined local E28 compact reports. Do not infer an
unbiased success rate from the quota-selected positive/negative population.

## Remaining throughput and bounded K2 trial

E28 took 53.38 minutes: opening 234.5 seconds, collection 685.9 and training/
publication 2270.4. Trainer timings include parent restore 45.8, training and
checkpoint 441.4, FP32 state export 992.1, and controller publication 577.3.
The current-parent cache already makes restore relatively small. Retaining the
authenticated sole-current model/state may avoid future rehydration, provided
ROOT ACK, exact parent descriptor and optimizer lineage are checked and obsolete
parents are retired. It cannot remove full R2 export and independent readback:
the next optimizer state must remain durably published before adoption.

Priorities are (1) remove per-document historical snapshot rewrites, (2) measure
new epoch captured/committed fraction and successful/deferred slots, (3) reduce
export/publication overhead while preserving full durable state and readback,
then (4) perform a bounded K2 comparison after stable calibration.

Use K=2,L=2 for that comparison: current admission pairs `zip(positives,
negatives)`, so K=2,L=1 still trains one pair and leaves the second positive
outside the objective. K=L=2 gives two pairs per task while preserving total
unit task weight. Start with 64 non-held-out tasks at one frozen checkpoint,
same task order and signed draw stream, compare K=L=1 against K=L=2 under the
same 16-attempt and 1,024-token limits and 600-second collection bound. Record
all attempt outcomes, repeated trajectories, quota completion/exhaustion,
output length/EOS cause, generation time, captured fraction and distinct tasks
versus pairs per wall hour. Require zero infrastructure/protocol failures and
complete original job evidence before training adoption. A higher pair count
with fewer completed tasks is not automatically higher useful throughput.
Follow any training trial with paired same-task held-out evidence; do not alter
precision, learning rate or objective simultaneously with K/L.
