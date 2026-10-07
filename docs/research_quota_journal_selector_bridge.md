# Captured-revision/native-outcome join (default off)

`ops.paired_quota_journal_selector_bridge.ResearchQuotaBridge` joins the research
revision journal and nested selector. It has no production caller and requires
explicit `enabled=True` to create/reopen its private database. No existing wire,
quota, miner, scoring, trainer, client or continuous audit path is changed.

Call `select_revision(journal, adapter, miner_public_key, manifest_sha256,
revision_id, native_outcomes, eos_token_ids=..., authenticate_original=...,
authenticate_admitted_native=...)`. Revision ID is explicit; the bridge never
chooses the latest head or substitutes an older/newer model. The caller must
freeze the actual revision and study plan before training.

The journal's new read-only `frozen_original(slot_id, revision_id)` returns the
first recorded ORIGINAL document with the exact stored batch, its original
capture evidence and scope-bound receipt. It checks the original full SHA and
canonical bytes inside a read transaction. The caller's original authenticator
then independently rechecks captured miner commitment/ROOT cheap-admission,
manifest/model/source/owner/task and exact original child SHA/size. Stored receipt
construction is not authentication. Adapter task/harness/draw and public-key
scope must match; the cumulative revision is recomputed from original bytes.
Wrapper repacks cannot shift the chosen original after it was recorded.

`native_outcomes` is a bounded list of exact `{attempt,evidence}` observations;
it cannot supply replacement rows or claimed member IDs. Rows are normalized
from the authenticated original batch and matched by prescribed attempt. A
captured original row missing a native outcome remains not-observed; an outcome
for an absent row can only be authenticated failure/indeterminate evidence, never
a selected graded trajectory. The native authenticator independently verifies
actual native grader evidence, complete rowSHA/task/checkpoint/epoch/owner/EOS
scope and returns the existing exact nested-selector receipt. This callback is
TRUSTED application code, not an assurance flag. This change supplies the durable
join and consistency checks, not an actual deployed signature/grader bridge.

All eligibility, EOS, binary reward/class consistency, execution/content/token
deduplication and complete-prefix rules reuse the nested selector. Indeterminate
or native-incomplete outcomes remain visible. Missing earlier attempts block
that arm's revision; authenticated explicit failed statuses count as observed;
missing attempts after a completed prefix can represent stopped collection.
Two pairs remain disjoint and nested, with total task weight unchanged.

The bridge database stores immutable context, authenticated native outcomes plus
original evidence and frozen selection packets. Context freezes the original
journal scope and tokenizer EOS selection scope. Per-attempt outcomes/evidence
cannot change across restart. New outcome packets are cumulative and cannot omit
known outcomes. Exact historical packet redelivery is a no-op and returns its
original historical result, even after a newer captured revision was processed;
it does not become the current head. A different packet for an older revision
must still retain and agree with known outcomes. Identical concurrent deliveries
commit one packet. A conflicting delivery refuses rather than overwriting it.

Results bind captured revision/original SHA, authenticated native inventory and
full selector result. Input assurance remains **cheap-eligible unaudited**; native
grading is reported separately and is not model-generation/TOPLOC proof. No live
inference-audit requirement is added. The bridge performs no optimizer work and
its packet identity is not an optimizer application identity. Original capture
and native join are separate metadata transactions; a crash can leave capture
committed with no join. Reauthenticating and rebuilding that metadata join is
safe because it neither regenerates trajectories nor reapplies training. It does
not make optimizer/checkpoint publication exactly once or install a live contract.

CPU controls use synthetic real Ed25519 signatures at both trusted boundaries:
restart/history/replay, immutable captured rows, checkpoint/task/owner/epoch
mismatches, native status changes, absent-row/native mismatch, complete prefixes,
content duplicates, original wrapper repacks, signature tamper/boolean refusals,
16 concurrent deliveries and actual subprocess exit78 after inserts before
commit. That crash leaves no orphan bridge outcomes/packets, preserves original
capture, and permits one metadata-only recovery. No model/native-grader/GPU
qualification is represented by these fixtures. Database triggers protect records
through this schema, not against arbitrary file or schema replacement.
