# Prospective native outcome eligibility for unaudited training

This default-off CPU proposal adds no controller hook or production admission.
The frozen f213 learner authenticates committed tokens, task/prompt context and
claimed positive/negative quotas; `validate_native_prompt` does not grade answers.
The full verifier grades native outcomes only after probability/TOPLOC/CDF checks.
A label guard therefore addresses a separate risk from sampling provenance.

`ops/native_training_outcome_filter.py` authenticates a ROOT-scoped context and
original learner documents, checks the approved source loader and exact runtime
source map, authenticates trusted task data and tokenizer assets, decodes the
original output tokens, then invokes the original isolated native MATH grader.
The original grader authenticates its Python/stdlib/dependency profile itself.
It receives trusted gold and decoded reply as JSON arguments, never miner code.
Each child has a 1 GiB address-space bound and finite CPU/wall bounds. At most
four children grade at once; 256 pairs, 512 grades, 600 seconds are hard limits.
The actual serialized JSON argument is capped at 96 KiB to allow JSON Unicode
escaping within Linux argument limits. Spawn failure, timeout, runtime refusal
and non-binary output are indeterminate, never negative samples.

A pair is retained only if both original claimed classes match exact native
0/1 outcomes. Any mismatch or indeterminate excludes the whole pair. Original
claims, documents, draws and audit population stay immutable. There is no label
swap, replacement draw, credit or cheating penalty. Matching native labels do
not certify tokens were sampled or proofs were verified. The receipt continues
to state `sampling_assurance=unaudited` and `proof_verification_performed=false`.

## Actual CPU evidence

The read-only E32 measurement predeclared every eighth of the original 256
learner-selected pairs before reading replies. It authenticated the original
ROOT job, all 177 executed f213 runtime files, original document commitments and
prompt eligibility, exact tokenizer assets and original native runtime profile.
All 64 grades matched their claims; no mismatch or indeterminate was found in
this 32-pair sample. Eleven negative replies ended at the 1024-token cap without
EOS and graded zero legitimately. Correct labels do not solve the truncation /
length confound or establish correctness of the unmeasured population.

Four-worker wall time was 11.099 seconds: 0.031 seconds for prepared-pair checking
and decoding and 11.068 seconds for concurrent native subprocesses. Their summed
child wall time was 42.715 seconds, maximum 0.838 seconds per grade. The parallel
sum is not elapsed wall time. Factory/source admission and original-object GETs
are outside these core timings. Linear projection is about 89 seconds for 256
pairs, not a full-population throughput measurement. All genuine tested grades
ran under the proposed 1 GiB child bound; whole-node peak RSS remains a rollout
qualification gate.

Seven additional actual public-entry controls used separately signed,
CPU-ONLY / NEVER-DISPATCH research documents derived from the original tokens.
Honest labels were accepted; reversing both classes excluded the pair without
altering the historical document; an unavailable native runtime was
indeterminate. Corrupt source/grader bindings, a changed document and a foreign
policy signer were refused. The signing authority was an ephemeral test key,
not ROOT. Eighteen portable CPU controls also cover argument inflation,
subprocess timeout/output, forged text, file links/tampering, bounds and
prevalidation before grader execution.

Private evidence lives under `state/root-audits/native-training-outcome-CPU-proposal-20261007-v1`:

- `PREDECLARED-E32-every8-32-pairs.private.json`
- `ACTUAL-E32-32-native-label-measurement.private.json` (original measurement)
- `ACTUAL-E32-32-native-label-measurement-with-costs.V2.private.json`
- `ACTUAL-public-boundary-CPU-controls.private.json`

## Proposed operator integration, requiring review

Keep remote f213 source, task-normalized objective, AdamW schema/settings and
optimizer lineage unchanged. Do not add fields to old manifests or alter an
already-signed training request. Existing `select_training_documents` journals
bind the original eligible inventory and original computation; the controller
also compares cached original job inputs exactly. Arbitrary caller-side
subsetting of an existing job would violate these commitments.

At a future unopened boundary, ROOT must explicitly authorize a new CPU
eligibility policy and operator overlay. A signed selection context should bind
original signed manifest/computation SHA, durable parent descriptor/global step,
full captured/selected document inventory and original selection journal SHA,
source/runtime/taskset/snapshot/tokenizer/grader digests and exact limits. It
must not be shaped as a dispatchable train request. The public function in this
proposal uses a signed never-dispatched job preview for CPU qualification; the
final operator hook should consume that non-dispatchable context instead.

The operator hook belongs before capacity/first job construction, at the
persistent controller's immutable learner-population handoff. It writes a new
append-only eligibility receipt and authorized accepted-subset journal without
overwriting original capture/population/selection files. Final training coverage
and the fresh ROOT-signed train request must bind exactly that accepted subset
and its eligibility receipt. Reattachment verifies the same receipt/subset and
reuses the same original job; it never regrades mutable replacements. No accepted
pairs means a typed no-update disposition, not an empty optimizer step.
Historical and future audit sampling continues over the full original eligible
population; excluded labels have no scoring/fraud side effect. Native grader
outage must not silently fall back to claimed labels. This is CPU native
eligibility, not an inference-audit barrier.

The operator route can preserve the scientific f213 source key and genuine
optimizer parent/cache without a cold source change or reset, if ROOT approves
these explicit CPU selection semantics and ordinary job/report lineage guards.
It changes the training data policy and requires actual pipeline/subset/restart
qualification before activation. A manifest-level scientific policy instead
requires a new complete source/admission/qualification, and its optimizer cache
source transition must be explicitly reviewed; no source alias or old ACK
relabeling is allowed.
