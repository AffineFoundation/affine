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

## Implemented preselection/controller candidate

`ops/native_training_eligibility.py` now implements the append-only context,
grade and accepted-subset journals and the optional controller selector.
`filter_eligibility_context` accepts a ROOT-signed **non-dispatchable** context
and a separate ROOT authorization. Neither context has a backend role/job ID,
training-output capabilities or optimizer mutation command. It binds the
original population and selection FILE SHAs, original signed computation,
full source map, immutable original document/admission inventory and actual
parent binding. A signed receipt is created before the first job signature.

The selector fetches at most four original documents concurrently, verifies
size/SHA, preserves original bytes, then grades under the original source.
It checks complete per-document/per-pair dispositions, exact binary scores and
class/status agreement rather than trusting an `accepted` flag. A whole
document is retained only if every pair matches. Derived coverage binds the
accepted original document inventory with the original seed/capture timestamp;
the original computation, optimizer parent and audit population are unchanged.
Restart revalidates the signed immutable context/grade/subset and reuses the
same subset without GET/regrading. An issued original job without a prior
eligibility receipt is refused: this is never an adoption/subsetting mechanism.

`prospective/native-training-controller-overlay/subnet/persistent_training_controller.py`
is the exact frozen f213 CPU controller plus one optional selector
call before capacity and dispatch. The remote 177-file scientific map remains
f213. A reviewed operator runner must install the selector on the controller
under a new explicit signed execution policy and pin both operator modules and
this CPU overlay; the hook defaults off when no selector exists. This candidate
has not been installed into any active runtime or config.

Fourteen additional controller/journal controls execute subset/all-accepted and
zero-accepted paths, restart/graded-receipt crash recovery, pre-existing-job
refusal, changed original/parent/receipt refusal, original-file preservation and
the actual frozen controller function up to its first intercepted dispatch.
Three actual non-dispatchable context controls use original E32 tokens under
an ephemeral CPU-only research signer: honest accepted, reversed labels
excluded, original runtime unavailable indeterminate. Their private receipt is
`ACTUAL-nondispatchable-context-CPU-controls.private.json`.

The whole CPU context process high-water RSS was 1,142,476 KiB. Child rusage
includes inherited pre-exec high-water; it is not a measurement of isolated
native-grader peak RSS. A future operator deployment must admit parent plus
four bounded children and qualify whole-node memory/CPU budget.

Zero accepted documents produce a durable signed `no_update` receipt and a
typed `NativeNoUpdate` **before any train dispatch**. The proposal deliberately
does not invent a successful training report or optimizer promotion. Ordinary
loop integration must separately review a no-update epoch disposition/closure
handler before activation; otherwise this typed condition stops dispatch rather
than silently admitting claimed labels. Likewise, local owned document copies
need ACK-based coded retirement under the final operator policy. These are
remaining activation gates, not deployed behavior or manual cleanup instructions.

The durable selector itself also ran three genuine-token CPU controls, with no
mocked grading: honest original tokens accepted, reversed original classes
produced `no_update`, and unavailable pinned native dependencies produced
indeterminate `no_update`. Each path preserved its original signed research
document/population/selection bytes and reused the same signed result on restart
without regrading. Receipt: `ACTUAL-native-selector-genuine-token-CPU-controls.private.json`.
The test issuer remained ephemeral; no production ROOT key or dispatch was used.

The controller hook explicitly refuses startup recovery combined with the native
selector until that distinct recovery context is authorized and graded; recovery
`apply` cannot replace final inputs after selection. A real controller-path
control demonstrates no apply/dispatch on this combination. An all-accepted
completed-metrics restart also reaches checked original report reuse without
new grading or dispatch.

Prospective durable runner integration (not deployed)

The optional `native_training_eligibility` row in a ROOT-signed durable learner
policy binds exactly two operator files, an independently signed preselection
authorization, the tokenizer directory/interpreter, and a future boundary. The
CPU overlay separately declares `subnet/persistent_training_controller.py` as an
operator exception. The baseline scientific source and all 177 remote runtime
pins remain unchanged. The authorization binds both the scientific baseline
`source_root` and the distinct CPU `execution_root`; the imported native prompt
module must retain the original scientific hash.

`prepare_runtime` imports the approved GPU service from the CPU overlay and
wraps its actual `RemoteController` constructor. The operator files load under a
private relative-import package, with file hashes checked before execution.
Default policies retain the original constructor. The future selector skips
rounds before the signed floor and every already-issued original training job
without an existing native subset journal. An issued native-selected job must
reuse and authenticate its original subset journal. New selection requires the
approved epoch prefix, source, contract fields and minimum optimizer parent
step. The prospective template uses earliest round 34 and minimum step 24;
these are gates, not claims that checkpoint 24 has already closed.

The CPU controller checks applicability before its startup-recovery guard, so
historical recovery observation remains unchanged. An applicable future
startup recovery is refused until separately authorized native grading of its
final inputs exists.

Deployment remains blocked on two explicit operational controls: an empty
selection must close with a signed **no-update** completion preserving the
checkpoint and optimizer pointer, with infrastructure indeterminacy distinguished
from conclusive label exclusions; and owned downloaded documents must retire
in software after genuine durable completion/ACK. No empty optimizer job,
optimizer reset, manufactured advancement, or manual file deletion is permitted.
The activation review must confirm the signed floor is still an unopened epoch;
current E33 and other issued requests remain immutable.

Qualification includes the actual `prepare_runtime` and real frozen
`RemoteController` constructor against the complete copied f213 CPU overlay,
an ephemeral CPU-test authority and isolated state. Only the remote metadata
query is stubbed. This is an import/constructor test, not a GPU or scientific
qualification. The production authority, queue, controller state and jobs are
not changed. New constructor tests cover private import binding, changed
operator bytes, source/contract/parent failures, old-issued-job bypass and
unauthenticated subset-journal refusal.

No-update and owned-copy lifecycle candidate

The prospective boundary is now round **35 or later**, after genuine E34
closure. E34 and its already-issued training request are untouched. The added
CPU GPU-service patch handles only the pinned operator's `NativeNoUpdate`
exception; every other exception propagates normally. It records a signed
no-update receipt, returns the exact approved checkpoint and optimizer pointer
with zero steps, and follows the existing after/completion path. Signed
completion states either `closed_native_label_exclusions_no_update` or
`closed_native_indeterminate_no_update`. The latter explicitly permits retry
only through a new unopened epoch using the same parent, without regrading or
reissuing the original epoch. Neither case claims training or cheating.

Each accepted-input manifest additionally binds the exact signed native
context, grade and subset receipt hashes before future job signature. These
metadata leave the original scientific computation projection, remote f213
177-file map, original capture population and audit/reward scoring unchanged.

Derived document copies now publish as atomic paired bundles: document bytes
and their ROOT-signed inode ownership become visible together using Linux
`renameat2(RENAME_NOREPLACE)`. A crash before ownership leaves an unowned
private staging orphan preserved; a fresh installation can proceed without
adopting or deleting that orphan. A final bundle never appears without its
ownership receipt. The CPU controls exercise both this failure and inode
continuity through the known directory rename.

A single default-off observer, enabled only by the signed native lifecycle
policy at actual execution, reconciles completed namespaces. It waits for a
signed epoch completion, the controller's advanced round, and no active
selector lease. It performs bounded full GET/SHA checks of the original frozen
R2 document keys, archives/readbacks the immutable signed native receipts and
completion, and persists a signed full-R2 ACK before any local unlink. Only the
originally owned derived document inode may retire; captured documents,
selection/population files, jobs, optimizer pointers and receipt history remain.
Per-copy signed unlink intents make crashes both before and after unlink
resumable. Failed GETs, changed inode/hardlink/ownership, changed ACK membership
and active leases refuse deletion. Preserved unowned staging orphans remain an
explicit exception, not a claim of retirement.

Focused controls now include the real copied f213 once-loop no-update closure,
immutable no-update restart, truthful infrastructure status, exact parent
preservation, full-R2 ACK and idempotent retirement, lease deferral, forged ACK,
inode replacement, and both unlink crash points. These are CPU controls;
production remains unchanged. Full 256-document native-grading budget and
memory qualification is a separate non-dispatchable original-E34 measurement
requiring its own expiring ROOT authorization/context. The prior 32-pair timing
is not substituted for that full workload result.

The full 256-document E34 benchmark exposed a token-bound defect before native grading: the tokenizer has 151,665 entries, while the original authenticated model config declares a 152,064-wide vocabulary. Ninety-one of the 512 authentic rollouts contain padded model vocabulary IDs. The corrected proposal binds a small local model config by the original checkpoint's `config.json` SHA and checks its declared integer vocabulary against the signed operator binding. It bounds tokens by that vocabulary and retains original tokenizer decoding, including padded IDs. Tokenizer length is checked for coverage, rather than used as the model's token limit. The failed benchmark and original inputs remain unchanged; corrected prevalidation is not a native-grading qualification or sampling proof.

The prospective CPU policy may opt in with
`limits.terminal_rule = "max-or-eos-v1"`. This preserves the interpretation of
older policies without that field. For each original output, the filter uses
the authenticated tokenizer EOS ID and the original signed environment/harness
output cap. It excludes a pair if an output stops before the cap without EOS,
or contains EOS before its final token. Miner-provided `done`, finish reason,
cap, text, and EOS metadata cannot override this rule. An excluded pair runs no
native grader, retains its original claims, and receives no cheating penalty.
The signed receipt records `excluded_terminal_rule`; zero-selection closure
counts these as known eligibility exclusions, not unavailable-grader retries.

This is terminal framing only. EOS at the end does not establish that EOS was
selected by the prescribed stream; public CDF, prompt, task, and seed checks
remain independently audited. A full-length non-EOS negative can remain a valid
native-negative example without being a useful completed mathematical answer.
Production requires a distinct ROOT-approved operator/authentication policy and
future unopened boundary; existing signed jobs and pinned runtimes are unchanged.
